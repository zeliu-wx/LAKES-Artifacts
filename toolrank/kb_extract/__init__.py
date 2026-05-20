"""Modular knowledge-base extraction pipeline for ToolRank."""


def run_kb_extract_pipeline(*args, **kwargs):
    from toolrank.kb_extract.pipeline import run_kb_extract_pipeline as _run_kb_extract_pipeline

    return _run_kb_extract_pipeline(*args, **kwargs)


__all__ = ["run_kb_extract_pipeline"]
