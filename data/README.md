# Data preparation code

Python modules live directly in this directory. The import package is `data`;
the distribution and installed command are named `hojil-data`.

From the repository root:

```bash
python -m data --help
python -m unittest discover -s tests -v
```

Install optional processing dependencies with
`pip install -e '.[audio,parquet,neural]'`. Models are provisioned explicitly;
commands never download them implicitly. Store recordings, datasets, weights,
and generated artifacts outside Git.

See the [documentation index](../docs/README.md),
[source review](../docs/1-open-source-data-review.md),
[preparation strategy and results](../docs/2-data-preparation.md), and
[command reference](../docs/data-pipeline.md).
