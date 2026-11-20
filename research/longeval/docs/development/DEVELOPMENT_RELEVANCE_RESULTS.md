# Development relevance analysis

The 129 existing ranking conditions have been scored on the **50 development queries**
against both DCTR and raw training judgments: 258 condition/source rows and 12,900
per-query rows. Every nDCG@10 and relevant-document Recall@100 value agrees with
`pytrec_eval` to absolute tolerance 1e-12. This includes the three exact references,
108 original HNSW runs, and 18 additional work-budget/wide-search runs.

**All judgments used here belong to snapshot 1.** Only snapshot-1 comparisons use
contemporaneous labels. Later scores use the same historical labels; unjudged added
documents are not evidence of irrelevance. These results are not a completed temporal
effectiveness study. No test-query retrieval outcomes were inspected or scored.

## Final-snapshot illustration using historical DCTR labels

Exact nDCG@10 is 0.266470 and relevant-document Recall@100 is 0.625915.

| Epzero incremental seed | ef=200 nDCG@10 | Offline 4,000-work nDCG@10 | Native ef=3200 nDCG@10 | 4,000-work ANN recall@100 |
|---:|---:|---:|---:|---:|
| 11 | 0.238902 | 0.257788 | 0.266523 | 0.9618 |
| 29 | 0.257788 | 0.257788 | 0.266523 | 0.9668 |
| 47 | 0.242584 | 0.242584 | 0.266523 | 0.9668 |

For seed 11, the nDCG gap below exact shrinks from 0.027568 to 0.008682 in the
4,000-work condition. Seed 47 does not obtain that nDCG improvement. Seeds 29 and 47
have identical mean ANN recall at 4,000 evaluations but different relevance scores.
Consequently, meeting the mean ANN target alone does not establish relevance equivalence.
These comparisons are descriptive, share queries, and are not significance claims.

At native ef=3200 the nDCG score slightly exceeds exact. This is permitted: exact
embedding similarity does not maximize judged relevance. Report signed ANN-minus-exact
differences rather than clipping improvements to zero or assuming an effectiveness ceiling.

The 4,000-work rankings are offline prefixes of ef=3200 traces. Compared with native
ef=200, both beam capacity and work differ. This is neither an equal-cost comparison nor
a measured latency gain. The selected-query rescue experiments used other work budgets;
their 99% neighbor recovery must not be substituted into this table.

## Across snapshots

Exact DCTR nDCG@10 against the fixed historical labels is 0.297205, 0.274965, and
0.266470. This decline includes competition from added documents under old judgments;
it cannot establish changing relevance, encoder drift, or causal shares of temporal loss.
The final study must use snapshot-specific judgments for the separate test queries.

The full raw-click and DCTR results, both construction backends, all three modes and
seeds, both original effort regimes, and the additional search conditions are retained.
There is no outcome-based selection of reported queries or omission of target misses.

## Reproduction and evidence

```bash
python -m research.longeval.relevance_analysis build/longeval-development-relevance-NEW
```

Executed output: `build/longeval-development-relevance-02/`:

- `report.json`: exact/run/verification/qrel input hashes, limits and all summaries.
- `summary.csv`: 258 condition/source rows, signed exact-reference differences.
- `per-query.csv`: 12,900 rows including judged coverage and neighbor recall.

`research/longeval/tests/test_relevance_analysis.py` checks independent evaluator parity
on a case where ANN improves judged relevance while losing a true vector neighbor, and
rejects incomplete populations and duplicate documents. Two tests pass on Windows.

The work-budget analysis, including its previous status document, was preserved before
advancing status in `build/longeval-work-budget-analysis-01.tar.gz`, SHA-256
`672e46c8ce3ec8f1c6b3285befe3da389e0c2995b9c417a9256cef996127e47a`.
