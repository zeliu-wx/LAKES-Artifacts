"""Project scheduling evidence from PaperDossier into passage_store.json."""

from __future__ import annotations

import json
from pathlib import Path

from toolrank.kb_extract.models import PaperDossier
from toolrank.schemas_v2 import Passage, PassageStore


def project_dossier_to_passages(
    dossier: PaperDossier,
    allowed_tool_ids: set[str] | None = None,
) -> list[Passage]:
    """Return owner-oriented passages stored on working memory."""
    if not dossier.working_memory or not dossier.working_memory.scheduling_evidence:
        return []

    passages = list(dossier.working_memory.scheduling_evidence)
    if allowed_tool_ids is not None:
        passages = [
            passage
            for passage in passages
            if passage.owner_tool in allowed_tool_ids
        ]
    return passages


