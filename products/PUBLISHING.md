# Publication surface

The authoritative publication surface is versioned:

- `published/<run_id>/` holds one complete, hash-verified release per
  promoted run, with its `publication_manifest.json` (run id, as-of, code
  version, per-file sha256);
- `latest` is the single authoritative pointer, switched atomically only
  after a release is fully staged and verified.

Each successful forecast publication includes
`peru_gdp_model_paths.csv`. It freezes S1 and the S2 recursive-AR(1)
challenger at the publication as-of, before any future GDP outcome is read.
The two models share the official node-1 nowcast and the current production
width curve, so later scoring isolates the effect of changing the center.
Historical records remain under their versioned `published/<run_id>/`
directories even after `latest` advances.

Pointer-failure recovery: if the atomic pointer switch itself fails, the
previous `latest` stays authoritative, the temporary pointer is removed, and
the new (valid but unreferenced) release is moved aside as
`published/.unreferenced-<run_id>-<stamp>`. Fix the cause and simply
republish the run; the quarantined directory is inert evidence and can be
deleted once superseded.

The old flat generated files that used to live directly in this directory
were REMOVED on 2026-08-06 (they had been frozen since the last flat
publication). The ONLY generated surface is `published/` plus the `latest`
pointer; downstream consumers must read `latest/`. The Python
files here are the assembly package's source code and are unrelated to
publication.
