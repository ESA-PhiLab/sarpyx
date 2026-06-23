# Evidence

- Package metadata was 1.0.1 while README, conda, citation, and submodule versions were older.
- Internal link check found missing developer guide, stale example/tutorial links, and missing generated notebook targets.
- `docs/build_site.py` failed under system Python because `markdown` was absent from the managed dev dependency set.
