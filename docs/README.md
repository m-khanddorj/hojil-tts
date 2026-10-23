# Documentation

Start with these two documents:

1. [Open-source data review](1-open-source-data-review.md) catalogs the available sources, overlapping distributions, and license evidence. It preserves the original research snapshot dated October 21, 2026.
2. [Data preparation](2-data-preparation.md) explains the cleaning strategy, the actual source inputs, review-driven corrections, and the final corpus results.

For implementation and reproduction:

- [Pipeline reference](data-pipeline.md): acquisition, configuration, CLI commands, splits, audits, and export contracts.
- [Cleaning research and decisions](data-cleaning-research.md): evidence behind the enhancement and trimming choices.
- [Initial pilot](data-pilot-2026-10-22.md): historical small-sample experiments before the full preparation and later corrections.
- [Preparation code](../data/README.md), [example configuration](../configs/example.json), and [verification dependencies](../requirements-verification.txt).

The current scope is data preparation. Training, deployment, and model-quality
claims are outside this change. Recordings, model weights, local listening pages,
and generated reports remain outside Git; the preparation document records their
measured outcomes and provenance.
