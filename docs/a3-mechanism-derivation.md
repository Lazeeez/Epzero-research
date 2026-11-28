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

- **An instrumented build** (laptop, the 300,000-vector SIFT sample): record every layer-0 link's creation step and the
  step it is pruned, to measure survival against link age and against its distance rank in the list. This tests
  the reading above and gives the displacement rate as a function of age instead of a constant.
- **Out-of-sample predictions, recorded now** for a setting never built in this project's measurements of S and
  R: M 32, efConstruction 200 (M₀ = 64), the 300,000-vector SIFT sample of M1, shuffled, seeds 11 and 29, with k̄ as measured in those
  builds and no parameter fitted to them:
  1. C(i) within 3% of k̄ ln(n/i) in every tenth.
  2. R within 1 link of k̄(i/n)^(k̄/65) in tenths 3-10, and below it in the first tenth.
  3. S within 1 link of the formula with M₀ = 64 in tenths 3-10, and below it in the first tenth.
  4. Pruning reaches fewer nodes than at M 16: the measured S/C is at least 0.95 from the third tenth on.

## 5. Result of the M 32 predictions (28 November 2026)

Two builds (`research/longeval/accumulation.py run --M 32 --ef-construction 200`, Epzero 0.5.0 from
`~/epzero-dev-venv` as in M1, the 300,000-vector SIFT sample, shuffled, seeds 11 and 29; 112 s each). Scored by
`research/longeval/a3_model.py`; aggregates `research/longeval/a3-m32.json`. Each insertion chose 16.03
neighbours on average. Both seeds agree to 0.1 links in every tenth; seed 11:

| | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C, model (k̄ ln(n/i)) | 52.9 | 30.7 | 22.3 | 16.9 | 12.8 | 9.6 | 6.9 | 4.6 | 2.6 | 0.8 |
| C, measured | 49.9 | 31.0 | 22.9 | 17.5 | 13.3 | 10.0 | 7.2 | 4.9 | 2.8 | 0.9 |
| S, model | 51.3 | 30.7 | 22.3 | 16.9 | 12.8 | 9.6 | 6.9 | 4.6 | 2.6 | 0.8 |
| S, measured | 42.0 | 29.9 | 22.6 | 17.4 | 13.3 | 10.0 | 7.2 | 4.9 | 2.8 | 0.9 |
| R, model | 7.3 | 10.0 | 11.4 | 12.4 | 13.2 | 13.8 | 14.4 | 14.9 | 15.4 | 15.8 |
| R, measured | 5.8 | 10.3 | 12.6 | 14.0 | 15.0 | 15.6 | 16.1 | 16.6 | 16.8 | 17.0 |

**Predictions.** (1) C within 3% in every tenth: **failed**, both seeds (6% low in the first tenth, 3-8% high
after). (2) R within 1 link in tenths 3-10: **failed** (1.2-1.6 links above the model in tenths 4-10); below
the model in the first tenth: **held**. (3) S within 1 link in tenths 3-10 and below it in the first tenth:
**held**, both. (4) S/C at least 0.95 from the third tenth: **held** (0.988-1.000).

**Diagnosis (after the result).** The number of neighbours an insertion keeps grows over the build, from 13.1 in
the first tenth to 17.1 in the last at M 32 (10.6 to 12.3 at M 16), and the model used one mean. Exposure computed
from each insertion's own choices, Σ_{j>i} k_j/(j − 1), matches C to within 1% (49.6 against 49.9 in the first
tenth): exposure holds; the constant k̄ was the simplification that failed. With each node's own k and the later
arrivals' mean k in the displacement rate, R is still 1.0-1.6 links above random displacement in tenths 2-5. So
at both M the selection heuristic departs from random displacement in the same direction: it removes more of the
oldest nodes' incoming links (S in the first tenth: 26 against 34 at M 16, 42 against 51 at M 32) and fewer of
the newer nodes' reverse links than random displacement would. That is what dropping the longest, oldest links
first would do; the instrumented build (section 4) is the test.

**The model's status for the paper.** Exposure, with each insertion's own number of choices, is the term that
makes the newest nodes the least linked; it held at M 16 and M 32 without a fitted parameter. Random displacement
gives the size of the pruning term to within about a link except in the oldest fifth, where pruning is stronger
than random, and it is stated as an approximation.

## 6. Which links pruning removes: design (recorded 28 November 2026, before the build)

**Build.** M1's SIFT sample (300,000 vectors, L2), shuffled, seed 11, M 16, efConstruction 100, Epzero 0.5.0
uncompressed and single-threaded, as M1. After each insertion j, the layer-0 lists of j and of every node it
chose are read and compared with their previous state, which gives each link's creation step and the step it is
pruned. A **pruning event** is an insertion that removes at least one link from a chosen node's list, or whose
reverse link is refused. A random 2% of pruning events (seed 2026) are recorded with every candidate in the
list: its age (steps since the link was made), its squared distance to the list's owner, and whether it was
removed. Code: `research/longeval/a3_links.py`.

**Measures.** Within each event, the age rank and the distance rank of each candidate (0 youngest or closest, 1
oldest or farthest); the removed candidates' mean ranks; the mean within-event Spearman correlation of age and
distance; a logistic model of removal on age rank alone, distance rank alone and both; the share of events that
remove exactly one link.

**Predictions** (from the reading in section 5):

1. Removed links are older than kept ones: their mean within-event age rank is above 0.6.
2. Removed links are longer: their mean within-event distance rank is above 0.6.
3. Older links are longer: the mean within-event Spearman correlation of age and distance is above 0.1.
4. Age acts through length: with both ranks in the model, the age coefficient is less than half its value alone.
5. At least 80% of pruning events remove exactly one link.

**Seen before the full build (28 November 2026).** A pipeline test on the first 20,000 insertions (event rate
5%) checked the tracking (no untracked link; every list matched the tracking at the end) and showed that most
pruning events remove 7 links or more (3,252 of 3,348), against prediction 5. The predictions above are unchanged
and are scored on the full build as recorded.
