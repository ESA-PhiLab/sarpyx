# sarpyx Documentation

This directory contains the source documentation and generated static site assets for `sarpyx`.

## Start Here

- [Installation](user_guide/installation.md)
- [CLI usage examples](user_guide/cli_examples.md)
- [User guide](user_guide/README.md)
- [Tutorials](tutorials/README.md)
- [Examples](examples/README.md)
- [API reference](api/README.md)
- [Developer guide](developer_guide/README.md)

The published site is available at <https://esa-philab.github.io/sarpyx/>.

## Build the Static Site

```bash
python docs/generate_static_site.py
```

For the markdown-driven local site:

```bash
python docs/build_site.py
python -m http.server 8000 -d docs/_site
```
