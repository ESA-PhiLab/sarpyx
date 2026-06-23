# Lessons

- Keep docs builder runtime dependencies in the dev group; a transitive lock entry is not enough for `uv run --group dev`.
- Generated `_site` needs explicit copy rules for linked notebook and static report assets.
