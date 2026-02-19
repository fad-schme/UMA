from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    source_path: str
    source_hash: str
    pages: List[Tuple[int, str]]  # (page_num, text)
    extracted_at: datetime


@dataclass(frozen=True)
class NormalizedSection:
    section_id: str
    doc_id: str
    text: str
    page_range: Tuple[int, int]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_range: Tuple[int, int]
    position: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    paragraph_index_start: Optional[int] = None
    paragraph_index_end: Optional[int] = None


@dataclass(frozen=True)
class IngestReport:
    doc_id: str
    chunks_created: int
    facts_created: int
    graph_edges_created: int
    warnings: List[str]


@dataclass(frozen=True)
class IngestConfig:
    chunk_size_tokens: int = 400
    overlap_tokens: int = 150
    embed_batch_size: int = 16
    embed_max_retries: int = 3
    embed_initial_delay_s: float = 0.5
    embed_backoff_factor: float = 2.0
    embed_max_delay_s: float = 8.0
    extract_max_chunks: Optional[int] = None
    allow_empty_pages: bool = False
    doc_min_fact_words: int = 10
    doc_summary_enabled: bool = True
    doc_summary_max_facts: int = 5
    doc_episode_enabled: bool = True
    graph_update_concurrency: int = 8
    fact_extraction_batch_enabled: bool = False
    fact_extraction_batch_size_chunks: int = 4
    fact_extraction_batch_max_chars: int = 12000
