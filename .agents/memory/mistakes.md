# Mistakes

## Hidden `.agents` discovery

- Mistake: used `rg --files` without `--hidden`, missed existing `.agents/iterations/*`, and initially wrote iteration notes into an occupied iteration number.
- Prevention: when working under `.agents`, always inspect with `rg --files --hidden .agents ...` before choosing iteration numbers or paths.
