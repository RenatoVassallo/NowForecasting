"""The NowForecasting production run pipeline.

One entry point (`python -m pipeline.main`) runs, in order: (1) data update/snapshot,
(2) satellite nowcasts, (3) domestic nowcast, (4) plots + final report - each
saving a full vintage of artifacts under ``runs/<run_id>/``. Switches live in
``pipeline/config/params.py``; model specs in ``pipeline/config/metadata.py``.
"""
