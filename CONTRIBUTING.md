# Contributing

## Setup

```bash
git clone https://github.com/didayolo/ranky.git
cd ranky
pip install -e ".[dev]"
```

This installs ranky in editable mode, with the tools used for testing, building the docs and releasing.

## Tests

```bash
pytest
```

Tests live in `tests/`, one file per module. Docstring examples (`>>>`) are run as well. The GitHub Actions workflow (`.github/workflows/tests.yml`) runs the tests on several Python versions for each push and pull request.

## Documentation

The documentation website is generated from the docstrings with [pdoc3](https://pdoc3.github.io/pdoc/), and served by GitHub Pages from the `docs/` folder of the `master` branch. To update it:

```bash
pdoc --html --force -o html ranky
cp html/ranky/*.html docs/
```

Then commit the `docs/` folder. The website is updated a few minutes after the push.

Docstrings follow the Google style (`Args:`, `Returns:`). The module docstring of `ranky/__init__.py` is the home page of the documentation.

## Release

1. Update `__version__` in `ranky/__init__.py` (the package version is read from there) and add a section to `CHANGELOG.md`.
2. Run the tests: `pytest`.
3. Regenerate the documentation (see above).
4. Commit, tag and push:
   ```bash
   git commit -am "Version X.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```
5. Build and upload to PyPI:
   ```bash
   rm -rf dist/
   python -m build
   twine check dist/*
   twine upload dist/*
   ```
   `twine upload` asks for a PyPI API token (username `__token__`). You can also store it in `~/.pypirc`.
6. Optionally, create a release on GitHub from the tag, with the changelog entry as description.
