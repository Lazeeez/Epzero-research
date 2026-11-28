# A3: the arrival bias derived

For paper A (`plan-paper-a.md`). Started on 28 November 2026. The first part is a derivation checked against
screen M1's measurements, which existed before it was written: it is a fit to known data, not a prediction.
The predictions it makes for settings never measured are recorded at the end, before any build of them.

## Setting

Layer 0 of an HNSW graph, built by inserting n nodes one at a time. Each insertion j chooses k̄ neighbours
among the existing nodes (k̄ ≈ 12 at M 16, efConstruction 100: the selection heuristic keeps fewer than M),
links to them, and each chosen node x adds the reverse link x → j to its own list. A list holds at most
M₀ = 2M links; when a reverse link would exceed that, x's list is pruned back to at most M₀.

The in-degree of node i splits into S(i), links from later nodes j > i that chose i and still keep it, and R(i),
reverse links from earlier nodes that i chose and that still keep i.

## 1. Exposure (choices by later arrivals)

If the nodes are exchangeable, as in a shuffled build, insertion j chooses any given existing node with
probability k̄/(j − 1). The expected number of later choosers of node i is therefore

  C(i) = Σ_{j>i} k̄/(j − 1) ≈ k̄ ln(n/i).

M1 measured C within about 1% in every tenth of the shuffled builds (no preferential term).

## 2. Mean-field pruning: random displacement

Suppose that once a list is full, each further link displaces one of the M₀ + 1 candidates at random, so that a
link already in a full list survives one addition with probability M₀/(M₀ + 1) ≈ e^(−1/(M₀+1)).

**Reverse links R(i).** Node i's k̄ reverse links sit in the lists of older nodes, which are full by the time i
arrives. Each of those lists then receives one addition per later chooser, Σ_{j>i} k̄/(j − 1) ≈ k̄ ln(n/i) in
expectation, whatever the older node's age. So

  R(i) ≈ k̄ · exp(−k̄ ln(n/i)/(M₀ + 1)) = k̄ · (i/n)^λ,  λ = k̄/(M₀ + 1).

**Chosen links S(i).** Insertion j's own list starts with k̄ links and receives k̄ ln(n/j) additions over the rest
of the build; it fills after M₀ − k̄ of them, at t* = (M₀ − k̄)/k̄ in units of ln(n/j). A link j → i survives with
p(j) = exp(−max(0, k̄ ln(n/j) − (M₀ − k̄))/(M₀ + 1)), and with t = ln(n/j), T = ln(n/i),

  S(i) = k̄ ∫₀^T p(t) dt = k̄T for T ≤ t*, and k̄ [t* + (1 − e^(−λ(T − t*)))/λ] beyond.

Pruning therefore touches only the links of nodes in roughly the first n·e^(−t*) of the build (about the first
20% at M 16), and in(i) = S(i) + R(i) is about 2.4 k̄ for the oldest and about k̄ for the newest.

## 3. Against M1's measurements (no fitted parameter; k̄ as measured, M₀ = 32)

Per insertion tenth, shuffled builds (M1's table), model above and measured below:

| | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LongEval S, model | 34.1 | 23.1 | 17.0 | 12.8 | 9.8 | 7.3 | 5.3 | 3.5 | 2.0 | 0.6 |
| LongEval S, measured | 26.7 | 20.2 | 15.8 | 12.4 | 9.6 | 7.2 | 5.2 | 3.5 | 1.9 | 0.6 |
| LongEval R, model | 3.8 | 6.0 | 7.3 | 8.3 | 9.1 | 9.8 | 10.4 | 11.0 | 11.5 | 12.0 |
| LongEval R, measured | 2.7 | 5.2 | 6.8 | 8.0 | 9.1 | 9.9 | 10.6 | 11.1 | 11.5 | 11.8 |
| SIFT S, model | 33.7 | 22.6 | 16.6 | 12.5 | 9.5 | 7.1 | 5.1 | 3.4 | 1.9 | 0.6 |
| SIFT S, measured | 26.0 | 20.0 | 15.8 | 12.5 | 9.6 | 7.3 | 5.2 | 3.5 | 2.0 | 0.6 |
| SIFT R, model | 3.8 | 6.0 | 7.2 | 8.1 | 8.9 | 9.6 | 10.2 | 10.7 | 11.2 | 11.7 |
| SIFT R, measured | 2.6 | 5.3 | 7.0 | 8.3 | 9.4 | 10.2 | 10.9 | 11.4 | 11.8 | 12.1 |

R is within 0.7 links in tenths 2-10 on both datasets; S within 0.4 in tenths 3-10. Both fail in the same place,
the oldest fifth: they lose more links than random displacement predicts (S 26-27 against 34 in the first tenth,
R 2.6-2.7 against 3.8). **Reading (a hypothesis, not tested):** the selection heuristic does not displace at
random. It keeps the closer and more diverse candidates, and the links into the oldest nodes were made while the
graph was sparse, so they are longer than the links that arrive later and are dropped first.

## 4. What is next

- **An instrumented build** (laptop, 200,000 SIFT vectors): record every layer-0 link's creation step and the
  step it is pruned, to measure survival against link age and against its distance rank in the list. This tests
  the reading above and gives the displacement rate as a function of age instead of a constant.
- **Out-of-sample predictions, recorded now** for a setting never built in this project's measurements of S and
  R: M 32, efConstruction 200 (M₀ = 64), SIFT 200,000, shuffled, seeds 11 and 29, with k̄ as measured in those
  builds and no parameter fitted to them:
  1. C(i) within 3% of k̄ ln(n/i) in every tenth.
  2. R within 1 link of k̄(i/n)^(k̄/65) in tenths 3-10, and below it in the first tenth.
  3. S within 1 link of the formula with M₀ = 64 in tenths 3-10, and below it in the first tenth.
  4. Pruning reaches fewer nodes than at M 16: the measured S/C is at least 0.95 from the third tenth on.
