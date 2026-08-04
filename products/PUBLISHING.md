# Publication surface

The authoritative publication surface is versioned:

- `published/<run_id>/` holds one complete, hash-verified release per
  promoted run, with its `publication_manifest.json` (run id, as-of, code
  version, per-file sha256);
- `latest` is the single authoritative pointer, switched atomically only
  after a release is fully staged and verified.

The flat generated files directly in this directory (block CSVs, fan CSV,
figures, report files) are DEPRECATED: nothing writes them any more, and they
are frozen at their last flat publication (2026-08-04, run
2026-08-04__140712). Downstream consumers must read `latest/`. The Python
files here are the assembly package's source code and are unrelated to
publication.
