// Screen M3: an independent implementation of FreshDiskANN-style Vamana insertion (Singh et al. 2021,
// Algorithms 1-3), for docs/screen-m3-vamana.md. Not DiskANN's code.
//
//   vamana_driver VECTORS.bin ORDER.bin QUERIES.bin OUT_DIR R L_BUILD ALPHA L_SEARCH[,L_SEARCH...] [SLACK FROZEN]
//
// Screen A4 (docs/screen-a4-diskann-choices.md) adds two of DiskANN's choices, both off by default: SLACK (a
// neighbour's list takes reverse links unpruned until it would exceed floor(SLACK * R), then is pruned back to R)
// and FROZEN (1: every search starts from an extra node at the data's mean vector, never returned or counted;
// 2: the extra node is a copy of the first inserted vector, as DiskANN's frozen point is when built from empty).
//
// VECTORS.bin, QUERIES.bin: uint32 rows, uint32 dim, then row-major float32. ORDER.bin: uint32 count, then
// the insertion sequence (row indices). Writes OUT_DIR/labels-L{w}.bin (int32 [queries x 100], row indices,
// best first), OUT_DIR/indegree.bin (int32 per row), OUT_DIR/outdegree.bin, and a JSON line on stdout.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Matrix {
    uint32_t n = 0, d = 0;
    std::vector<float> v;
    explicit Matrix(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(&n), 4) || !in.read(reinterpret_cast<char*>(&d), 4) || !n || !d)
            throw std::runtime_error("bad header: " + path);
        v.resize(size_t(n) * d);
        if (!in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4))) throw std::runtime_error("short file: " + path);
    }
    const float* row(uint32_t i) const { return v.data() + size_t(i) * d; }
};

static inline float l2(const float* a, const float* b, uint32_t d) {
    float s = 0;
    for (uint32_t i = 0; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

struct Graph {
    const Matrix& x;
    uint32_t R, L;
    float alpha;
    size_t cap;                   // a list is pruned when it would exceed this (R without slack)
    bool frozen;
    std::vector<float> centre;    // the frozen node's vector (node x.n)
    std::vector<std::vector<uint32_t>> out;
    std::vector<uint32_t> mark;   // visit stamps
    uint32_t stamp = 0;
    int64_t entry = -1;

    Graph(const Matrix& data, uint32_t r, uint32_t l, float a, float slack, bool frz, int64_t copy_of = -1)
        : x(data), R(r), L(l), alpha(a), cap(size_t(slack * float(r) + 1e-6f)), frozen(frz),
          out(data.n + (frz ? 1 : 0)), mark(data.n + (frz ? 1 : 0), 0) {
        if (cap < R) cap = R;
        if (frozen) {
            centre.assign(x.d, 0.0f);
            std::vector<double> sum(x.d, 0.0);
            for (uint32_t i = 0; i < x.n; ++i) for (uint32_t k = 0; k < x.d; ++k) sum[k] += x.row(i)[k];
            for (uint32_t k = 0; k < x.d; ++k) centre[k] = float(sum[k] / x.n);
            if (copy_of >= 0) std::copy(x.row(uint32_t(copy_of)), x.row(uint32_t(copy_of)) + x.d, centre.begin());
            entry = x.n;
        }
    }
    const float* vec(uint32_t i) const { return i == x.n ? centre.data() : x.row(i); }

    // Beam search from the entry: returns the best `keep` and fills `expanded` with every expanded node.
    std::vector<std::pair<float, uint32_t>> search(const float* q, uint32_t width, uint32_t keep,
                                                   std::vector<std::pair<float, uint32_t>>* expanded) {
        if (++stamp == 0) { std::fill(mark.begin(), mark.end(), 0); stamp = 1; }
        std::vector<std::pair<float, uint32_t>> pool;      // sorted ascending, size <= width
        std::vector<char> done;
        auto e = uint32_t(entry);
        pool.push_back({l2(q, vec(e), x.d), e});
        done.push_back(0);
        mark[e] = stamp;
        while (true) {
            size_t pick = pool.size();
            for (size_t i = 0; i < pool.size(); ++i) if (!done[i]) { pick = i; break; }
            if (pick == pool.size()) break;
            done[pick] = 1;
            uint32_t node = pool[pick].second;
            if (expanded) expanded->push_back(pool[pick]);
            for (uint32_t nb : out[node]) {
                if (mark[nb] == stamp) continue;
                mark[nb] = stamp;
                float dist = l2(q, vec(nb), x.d);
                if (pool.size() >= width && dist >= pool.back().first) continue;
                auto it = std::lower_bound(pool.begin(), pool.end(), std::make_pair(dist, nb));
                size_t at = size_t(it - pool.begin());
                pool.insert(it, {dist, nb});
                done.insert(done.begin() + long(at), 0);
                if (pool.size() > width) { pool.pop_back(); done.pop_back(); }
            }
        }
        if (pool.size() > keep) pool.resize(keep);
        return pool;
    }

    // RobustPrune: candidates as (distance to p, id), p itself excluded.
    std::vector<uint32_t> prune(uint32_t p, std::vector<std::pair<float, uint32_t>> cand) {
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.second == b.second; }), cand.end());
        std::vector<uint32_t> kept;
        std::vector<char> alive(cand.size(), 1);
        for (size_t i = 0; i < cand.size() && kept.size() < R; ++i) {
            if (!alive[i] || cand[i].second == p) continue;
            uint32_t star = cand[i].second;
            kept.push_back(star);
            for (size_t j = i + 1; j < cand.size(); ++j)
                if (alive[j] && alpha * l2(vec(star), vec(cand[j].second), x.d) <= cand[j].first) alive[j] = 0;
        }
        return kept;
    }

    void insert(uint32_t p) {
        if (entry < 0) { entry = p; return; }
        std::vector<std::pair<float, uint32_t>> visited;
        search(x.row(p), L, L, &visited);
        out[p] = prune(p, visited);
        for (uint32_t j : out[p]) {
            auto& list = out[j];
            if (std::find(list.begin(), list.end(), p) != list.end()) continue;
            list.push_back(p);
            if (list.size() > cap) {
                std::vector<std::pair<float, uint32_t>> cand;
                cand.reserve(list.size());
                for (uint32_t c : list) cand.push_back({l2(vec(j), vec(c), x.d), c});
                out[j] = prune(j, cand);
            }
        }
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 9 && argc != 11) throw std::runtime_error("usage: VECTORS ORDER QUERIES OUT R L_BUILD ALPHA L_SEARCH[,..] [SLACK FROZEN]");
        Matrix x(argv[1]), q(argv[3]);
        if (x.d != q.d) throw std::runtime_error("dimension mismatch");
        std::ifstream oin(argv[2], std::ios::binary);
        uint32_t count = 0;
        oin.read(reinterpret_cast<char*>(&count), 4);
        if (count != x.n) throw std::runtime_error("order length differs from the data");
        std::vector<uint32_t> order(count);
        oin.read(reinterpret_cast<char*>(order.data()), std::streamsize(count) * 4);
        std::string out_dir = argv[4];
        const float slack = argc == 11 ? std::stof(argv[9]) : 1.0f;
        const std::string frozen_mode = argc == 11 ? std::string(argv[10]) : "0";
        const bool frozen = frozen_mode == "1" || frozen_mode == "2";
        Graph g(x, uint32_t(std::stoul(argv[5])), uint32_t(std::stoul(argv[6])), std::stof(argv[7]), slack, frozen,
                frozen_mode == "2" ? int64_t(order[0]) : -1);
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < count; ++i) g.insert(order[i]);
        double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<int32_t> indeg(x.n, 0), outdeg(x.n, 0);
        size_t over = 0;
        size_t max_degree = 0, to_frozen = 0;
        for (uint32_t i = 0; i < g.out.size(); ++i) {
            max_degree = std::max(max_degree, g.out[i].size());
            over += g.out[i].size() > g.cap;
            if (i == x.n) continue;                        // the frozen node is not counted
            outdeg[i] = int32_t(g.out[i].size());
            for (uint32_t j : g.out[i]) { if (j == x.n) ++to_frozen; else ++indeg[j]; }
        }
        std::ofstream(out_dir + "/indegree.bin", std::ios::binary).write(reinterpret_cast<char*>(indeg.data()), std::streamsize(indeg.size()) * 4);
        std::ofstream(out_dir + "/outdegree.bin", std::ios::binary).write(reinterpret_cast<char*>(outdeg.data()), std::streamsize(outdeg.size()) * 4);
        std::stringstream widths(argv[8]);
        std::string w;
        std::ostringstream times;
        while (std::getline(widths, w, ',')) {
            uint32_t width = uint32_t(std::stoul(w));
            std::vector<int32_t> labels(size_t(q.n) * 100, -1);
            auto s0 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < q.n; ++i) {
                // With the frozen node one extra result is kept, so that 100 data vectors remain once it is dropped;
                // without it the search is M3's, unchanged.
                const uint32_t keep = frozen ? 101 : 100;
                auto best = g.search(q.row(i), std::max<uint32_t>(width, keep), keep, nullptr);
                size_t k = 0;
                for (const auto& b : best) {
                    if (b.second == x.n) continue;               // never return the frozen node
                    if (k == 100) break;
                    labels[size_t(i) * 100 + k++] = int32_t(b.second);
                }
            }
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
            std::ofstream(out_dir + "/labels-L" + w + ".bin", std::ios::binary)
                .write(reinterpret_cast<char*>(labels.data()), std::streamsize(labels.size()) * 4);
            times << (times.tellp() > 0 ? "," : "") << "\"" << w << "\":" << secs;
        }
        std::cout << "{\"build_seconds\":" << build << ",\"entry\":" << g.entry << ",\"over_degree\":" << over
                  << ",\"max_degree\":" << max_degree << ",\"slack_cap\":" << g.cap << ",\"frozen\":" << (frozen ? 1 : 0)
                  << ",\"links_to_frozen\":" << to_frozen
                  << ",\"search_seconds\":{" << times.str() << "}}" << std::endl;
        return over ? 3 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
