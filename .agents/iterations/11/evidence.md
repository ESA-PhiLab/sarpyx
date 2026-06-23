# Evidence

- Started on branch `v1.0.0` with uncommitted `1.0.1` hardening changes.
- No local or remote `1.0.1` branch existed, so a new branch was created from the current HEAD and kept the uncommitted diff.
- Current launchers shared SNAP userdir state by default; logs had shown duplicate jobs using the same output roots and same SNAP userdir.
- Zarr writer removed the final output and wrote directly to the final `.zarr` directory, so interruption could leave a final-looking partial store.
