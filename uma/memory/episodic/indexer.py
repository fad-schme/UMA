"""
episodic/indexer.py
===================

EpisodeIndexer
--------------

Transforms a set of working memory entries into a structured
Episode object + embedding vector.

Responsibilities
----------------
- Summarize working memory using an LLM
- Create an Episode model (id, timestamp, summary, raw transcript)
- Embed episode summary for retrieval indexing
- Return (Episode, embedding)

Coding Agent Instructions
-------------------------
- Keep prompt templates minimal and modifiable.
- Ensure indexer is asynchronous.
- Use semantic/episodic models defined in types_episode.py.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from uma.common.types import Episode
from uma.adapters.llm.controller import LLMCallContext, generate_text

logger = logging.getLogger(__name__)


class EpisodeIndexer:
    """
    LLM-powered episode builder.

    Parameters
    ----------
    llm : Any
        Should implement an async generate(messages, max_tokens) method.
    embedder : Any
        Should implement an async embed(text) -> List[List[float]].
    """

    def __init__(self, llm: Any, embedder: Any):
        self.llm = llm
        self.embedder = embedder
        logger.debug("EpisodeIndexer initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def build_episode(
        self,
        *,
        owner_type: str,
        owner_id: str,
        turn_entries: list[Any],
        wm_context: list[Any] | None = None,
    ):
        """
        Build a structured Episode from the current turn.

        Parameters
        ----------
        owner_type : str
        owner_id : str
        turn_entries : List[dict]
            The current turn only — [{"role": "user", ...}, {"role": "assistant", ...}].
            This is the content that gets summarized into the episode.
        wm_context : List[Any], optional
            Prior working-memory entries (WMEntry objects or dicts). Used as
            background context in the prompt so the LLM understands the
            conversation, but NOT summarized into the episode.

        Returns
        -------
        (Episode, embedding_vector)
        """
        try:
            turn_transcript = self._wm_to_transcript(turn_entries or [])

            # Build system prompt — include WM as background if available
            system_parts = [
                "Summarize the following conversation turn into a short episode memory. "
                "Limit to one concise paragraph."
            ]
            if wm_context:
                wm_transcript = self._wm_to_transcript(wm_context)
                system_parts.append(
                    f"\nPrevious conversation context (for reference only — do not summarize):\n{wm_transcript}"
                )

            summary_msgs = [
                {"role": "system", "content": "\n".join(system_parts)},
                {"role": "user", "content": turn_transcript},
            ]

            try:
                summary = await generate_text(
                    llm=self.llm,
                    messages=summary_msgs,
                    max_tokens=128,
                    ctx=LLMCallContext(op="episode_summarize", owner_type=owner_type, owner_id=owner_id),
                )
            except Exception:
                logger.exception("EpisodeIndexer: LLM summary failed.")
                summary = "Conversation summary unavailable."

            # Build episode model — turn_id comes from turn entries
            turn_id = None
            try:
                for ent in reversed(turn_entries or []):
                    md = ent.get("metadata") if isinstance(ent, dict) else getattr(ent, "metadata", None)
                    if isinstance(md, dict) and md.get("turn_id"):
                        turn_id = str(md["turn_id"])
                        break
            except Exception:
                turn_id = None

            ep = Episode(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                summary=summary,
                user_id=owner_id,
                raw=turn_transcript,
                tags=[],
                meta={"turn_id": turn_id} if turn_id else {},
                owner_type=owner_type,
                owner_id=owner_id,
            )

            # Embed summary
            expected_dim = getattr(self.embedder, "dimension", None)
            if not isinstance(expected_dim, int) or expected_dim <= 0:
                raise ValueError("EpisodeIndexer: embedder.dimension must be a positive integer")
            emb = await self.embedder.embed([summary])
            if not emb or not isinstance(emb, list) or not emb[0]:
                raise ValueError("EpisodeIndexer: embedder returned empty embedding.")
            embedding = emb[0]
            if not isinstance(embedding, list) or len(embedding) != expected_dim:
                got_dim = len(embedding) if isinstance(embedding, list) else None
                raise ValueError(
                    f"EpisodeIndexer: invalid embedding dim (expected={expected_dim} got={got_dim})"
                )

            return ep, embedding

        except Exception:
            logger.exception("EpisodeIndexer.build_episode failed.")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wm_to_transcript(self, entries: list[Any]) -> str:
        """
        Convert WM entries (WMEntry objects OR dicts) into a unified transcript.

        UMA allows episodic indexing to receive either:
            • WMEntry(role, content) objects
            • dicts of the form {"role": ..., "content": ...}

        The EpisodeMapper standardizes WMEntry → dict, but the final turn
        entries added by EpisodicCore are also dicts. This method must
        robustly handle both forms.

        Returns
        -------
        str
            Transcript text in the format:
                "user: hello\nassistant: hi"

        Notes
        -----
        • Missing or malformed entries are logged and skipped.
        • Unknown roles default to "user".
        • Unknown content defaults to "".
        """

        lines = []
        for ent in entries:
            try:
                # Case 1: dict-style entries
                if isinstance(ent, dict):
                    role = ent.get("role", "user")
                    text = ent.get("content", "")

                # Case 2: WMEntry-like objects
                else:
                    role = getattr(ent, "role", "user")
                    text = getattr(ent, "content", "")

                lines.append(f"{role}: {text}")

            except Exception:
                logger.exception(
                    "EpisodeIndexer._wm_to_transcript: invalid WM entry=%r", ent
                )
                continue

        return "\n".join(lines)
