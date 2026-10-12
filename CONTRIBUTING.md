# Contributing

Start with the three decisions in [the product specification](docs/product-specification.md). Keep changes limited to an explicit requirement or demonstrated failure. Broader architecture references do not expand automated verdict scope by themselves.

Install Python 3.11 or 3.12 and run `python -m pip install -e '.[dev]'` inside a virtual environment. Core tests do not download or require model weights. Run:

```text
python -m pytest -q
python -m ruff check src tests scripts
python -m build
```

Follow [the dataset guide](docs/dataset-guide.md) for evidence and examples. Include primary-source URLs, dates, applicability conditions and trade-offs. Mark synthetic examples and engineering inferences explicitly. Never mark a record expert-reviewed without an actual review. Keep variants together and evaluation labels out of retrieval.

Changes to evidence or model input rendering require regenerating training exports and rerunning relevant baselines. Add failure cases that verify externally visible behavior. Report real model experiments separately from tests with simulated responses.

Before opening a pull request, remove credentials, private architecture details, local runtime files and model weights. Include the problem, resulting behavior and validation performed.
