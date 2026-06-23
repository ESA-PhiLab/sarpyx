# Lessons

Operational hardening needs both pipeline-level validation and launcher-level isolation; fixing only the DIMAP payload path still leaves duplicate jobs able to race on final products and SNAP cache state.

Do not recursively copy SNAP userdirs by default; large auxdata trees should be explicitly shared or pre-staged, preferably read-only, and job-local userdirs should only hold writable per-run state.
