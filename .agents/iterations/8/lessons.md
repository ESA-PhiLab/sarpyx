# Lessons

Do not treat SNAP `StampsExport` as the last runnable stage for StaMPS. The declared pipeline should model the external `mt_prep_*` handoff explicitly and fail early with actionable environment errors.
