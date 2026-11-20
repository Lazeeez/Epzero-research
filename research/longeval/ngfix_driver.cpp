// Controlled adapter for NGFix commit 9acd987cf1c420069f8fd602c0d4e4aa81053558.
// Compile with upstream headers and AVX-512. This is a competitor, not our method.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "ngfixlib/graph/hnsw_ngfix.h"

template<class T> struct Matrix {
    uint32_t n = 0, d = 0;
    std::vector<T> values;
    explicit Matrix(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(&n), 4) ||
            !in.read(reinterpret_cast<char*>(&d), 4) || !n || !d)
            throw std::runtime_error("invalid matrix header: " + path);
        uint64_t count = uint64_t(n)*d;
        if (count > std::numeric_limits<size_t>::max()/sizeof(T) ||
            std::filesystem::file_size(path) != 8 + count*sizeof(T))
            throw std::runtime_error("invalid matrix length: " + path);
        values.resize(count);
        if (!in.read(reinterpret_cast<char*>(values.data()), count*sizeof(T)))
            throw std::runtime_error("short matrix read: " + path);
    }
    T* row(size_t i) { return values.data() + i*d; }
};

using Graph = ngfixlib::HNSW_NGFix<float>;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error(
            "usage: ngfix_driver base.fbin queries.fbin calibration_top200.ibin output_dir");
        // Queries are calibration first, then development. Only calibration GT is loaded.
        Matrix<float> base(argv[1]), queries(argv[2]);
        Matrix<int32_t> gt(argv[3]);
        if (base.d != queries.d || base.d % 4 || gt.d != 200 ||
            gt.n >= queries.n || base.n < 200)
            throw std::runtime_error("inconsistent input dimensions or roles");
        for (float value : base.values) if (!std::isfinite(value))
            throw std::runtime_error("nonfinite corpus vector");
        for (float value : queries.values) if (!std::isfinite(value))
            throw std::runtime_error("nonfinite query vector");
        for (size_t q = 0; q < gt.n; ++q) {
            std::set<int32_t> unique;
            double previous = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < gt.d; ++j) {
                int32_t id = gt.row(q)[j];
                if (id < 0 || uint32_t(id) >= base.n || !unique.insert(id).second)
                    throw std::runtime_error("invalid calibration neighbor IDs");
                double distance = 1.;
                for (size_t d = 0; d < base.d; ++d)
                    distance -= double(queries.row(q)[d])*base.row(id)[d];
                if (distance < previous - 1e-12)
                    throw std::runtime_error("calibration neighbors not ordered by double IP");
                previous = distance;
            }
        }
        std::filesystem::path output(argv[4]);
        if (!std::filesystem::create_directory(output))
            throw std::runtime_error("output must be new");
        std::ofstream timing(output / "queries.tsv");
        std::ofstream ranks(output / "rankings.tsv");
        std::ofstream stages(output / "stages.tsv");
        if (!timing || !ranks || !stages) throw std::runtime_error("cannot open output");
        timing << "variant\tquery\tef\trepeat\tseconds\tupstream_ndc\n" << std::setprecision(17);
        ranks << "variant\tquery\tef\trank\tid\tdistance\n" << std::setprecision(9);
        stages << "variant\tstage\tseconds\tserialized_bytes\n" << std::setprecision(17);
        std::string base_path = (output / "base.index").string();
        {
            auto start = Clock::now();
            Graph graph(ngfixlib::IP_float, base.d, base.n, 16, 48);
            // Serial insertion and fixed input order; no asynchronous stream pairing.
            for (size_t i = 0; i < base.n; ++i) graph.InsertPoint(i, 100, base.row(i));
            double seconds = elapsed(start);
            graph.StoreIndex(base_path);
            stages << "base\tbuild\t" << seconds << '\t' << std::filesystem::file_size(base_path) << '\n';
        }
        std::vector<std::string> names = {"base", "ngfix", "rfix", "ngfix_rfix"};
        std::vector<std::unique_ptr<Graph>> graphs;
        for (size_t variant = 0; variant < names.size(); ++variant) {
            graphs.emplace_back(new Graph(ngfixlib::IP_float, base_path));
            auto& graph = *graphs.back();
            auto start = Clock::now();
            for (size_t q = 0; q < gt.n; ++q) {
                if (variant == 1 || variant == 3) {
                    graph.NGFix(queries.row(q), gt.row(q), 100, 100);
                    graph.NGFix(queries.row(q), gt.row(q), 10, 10);
                }
                if (variant == 2 || variant == 3) graph.RFix(queries.row(q), gt.row(q), 10);
            }
            double seconds = elapsed(start);
            auto path = output / (names[variant]+".before.index");
            graph.StoreIndex(path.string());
            stages << names[variant] << "\trepair\t" << seconds << '\t' << std::filesystem::file_size(path) << '\n';
        }
        for (int ef : {100, 200, 400, 800}) for (size_t q = 0; q < queries.n; ++q) {
            // Rotate variant order; one warmup and three timed calls per condition.
            for (size_t offset = 0; offset < graphs.size(); ++offset) {
                size_t variant = (offset + q) % graphs.size();
                size_t ndc = 0;
                auto expected = graphs[variant]->searchKnn(queries.row(q), 100, ef, ndc);
                if (expected.size() != 100) throw std::runtime_error("incomplete ranking");
                std::set<unsigned> unique;
                for (const auto& [distance, id] : expected)
                    if (id >= base.n || !std::isfinite(distance) || !unique.insert(id).second)
                        throw std::runtime_error("invalid ranking");
                for (int repeat = 0; repeat < 3; ++repeat) {
                    ndc = 0;
                    auto start = Clock::now();
                    auto actual = graphs[variant]->searchKnn(queries.row(q), 100, ef, ndc);
                    double seconds = elapsed(start);
                    if (actual != expected) throw std::runtime_error("unstable native ranking");
                    timing << names[variant] << '\t' << q << '\t' << ef << '\t'
                           << repeat << '\t' << seconds << '\t' << ndc << '\n';
                }
                for (size_t i = 0; i < expected.size(); ++i)
                    ranks << names[variant] << '\t' << q << '\t' << ef << '\t' << i+1
                          << '\t' << expected[i].second << '\t' << expected[i].first << '\n';
            }
        }
        for (size_t variant = 0; variant < graphs.size(); ++variant)
            graphs[variant]->StoreIndex((output / (names[variant]+".after.index")).string());
        timing.close(); ranks.close(); stages.close();
        if (!timing || !ranks || !stages) throw std::runtime_error("output write failed");
        std::cout << "completed; external score, provenance and state verification still required\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
