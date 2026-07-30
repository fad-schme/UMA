from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .types import ParsedDocument

logger = logging.getLogger(__name__)


def _hash_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ----------------------------- HTML sanitization --------------------------

_STRIP_TAGS = {"script", "style", "noscript", "iframe", "object", "embed", "link", "meta", "svg"}
_INLINE_EVENT_RE = re.compile(r'^on\w+$', re.IGNORECASE)
_JS_HREF_RE = re.compile(r'javascript\s*:', re.IGNORECASE)
_DATA_URI_RE = re.compile(r'data\s*:', re.IGNORECASE)
_COND_COMMENT_RE = re.compile(r'<!--\[if.*?<!\[endif\]-->', re.DOTALL | re.IGNORECASE)


def _sanitize_html(html_text: str) -> tuple[str, dict[str, int]]:
    """
    Strip dangerous content from HTML and return (sanitized_text, counts).
    counts keys: scripts, iframes, event_handlers, js_urls, data_urls, cond_comments, svg
    """
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "HTML sanitization requires beautifulsoup4: pip install beautifulsoup4"
        ) from exc

    counts: dict[str, int] = {}

    # Strip conditional comments before parsing
    cond_hits = _COND_COMMENT_RE.findall(html_text)
    if cond_hits:
        counts["cond_comments"] = len(cond_hits)
        html_text = _COND_COMMENT_RE.sub("", html_text)

    soup = BeautifulSoup(html_text, "html.parser")

    # Remove block-level dangerous tags
    for tag_name in _STRIP_TAGS:
        hits = soup.find_all(tag_name)
        if hits:
            category = "scripts" if tag_name in ("script", "noscript") else \
                       "iframes" if tag_name == "iframe" else tag_name
            counts[category] = counts.get(category, 0) + len(hits)
            for t in hits:
                t.extract()

    # Strip inline event handler attributes and dangerous URL schemes from all remaining tags
    for tag in soup.find_all(True):
        attrs_to_remove = []
        for attr, val in list(tag.attrs.items()):
            val_str = val if isinstance(val, str) else " ".join(val) if isinstance(val, list) else ""
            if _INLINE_EVENT_RE.match(attr):
                attrs_to_remove.append(attr)
                counts["event_handlers"] = counts.get("event_handlers", 0) + 1
            elif _JS_HREF_RE.search(val_str):
                attrs_to_remove.append(attr)
                counts["js_urls"] = counts.get("js_urls", 0) + 1
            elif _DATA_URI_RE.search(val_str):
                attrs_to_remove.append(attr)
                counts["data_urls"] = counts.get("data_urls", 0) + 1
        for attr in attrs_to_remove:
            del tag[attr]

    return soup.get_text(separator="\n", strip=True), counts


# ----------------------------- Strategy base -----------------------------


class ParserStrategy:
    """Synchronous parser returning a text string."""

    def read(self, file_path: str) -> str:
        raise NotImplementedError

    def read_pages(self, file_path: str) -> list[tuple[int, str]]:
        """Return per-page text. Default is a single page from read()."""
        return [(1, self.read(file_path))]


# ----------------------------- Parsers -----------------------------------


class TXTParser(ParserStrategy):
    """Detect encoding and return UTF-8 text."""

    def read(self, file_path: str) -> str:
        try:
            import charset_normalizer
        except Exception as exc:
            raise ImportError(
                "TXTParser requires charset-normalizer: pip install charset-normalizer"
            ) from exc
        try:
            match = charset_normalizer.from_path(file_path).best()
            if not match:
                raise ValueError("Failed to detect encoding")
            logger.debug(
                "TXTParser: '%s' encoding='%s' confidence=%.3f",
                file_path,
                match.encoding,
                getattr(match, "confidence", 0.0) or 0.0,
            )
            return str(match)
        except Exception as exc:
            logger.error("TXTParser failed for '%s': %s", file_path, exc)
            raise


class PDFParser(ParserStrategy):
    """Parse PDFs page-by-page; refuse files declaring excessive page counts.

    H2 defense: PyPDF2 allocates per-page during traversal, so a small file
    claiming millions of pages amplifies into memory exhaustion. The
    max_pages constructor argument bounds this — a PDF declaring more pages
    than the cap raises ValueError before any text extraction runs.
    """

    # Default cap when constructed without an explicit max_pages. Conservative
    # enough to cover any legitimate document; small enough to refuse the
    # "PDF claiming 10 million pages" class of attack. Override at parser
    # construction time via parse_file(..., pdf_max_pages=N).
    DEFAULT_MAX_PAGES = 5000

    def __init__(self, *, max_pages: Optional[int] = None) -> None:
        if max_pages is None or max_pages <= 0:
            self.max_pages = self.DEFAULT_MAX_PAGES
        else:
            self.max_pages = int(max_pages)

    def _open_reader(self, file_path: str):
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            raise ImportError("PDFParser requires PyPDF2: pip install PyPDF2") from exc
        reader = PdfReader(file_path)
        # Page-count guard: len(reader.pages) is O(1) on PyPDF2 and runs
        # before any per-page allocation. Refuse the file if it declares
        # more pages than the cap.
        try:
            declared_pages = len(reader.pages)
        except Exception:
            # If the reader can't even report a page count, treat as
            # malformed and refuse rather than risk an unbounded traversal.
            raise ValueError(f"PDFParser: cannot determine page count for '{file_path}'")
        if declared_pages > self.max_pages:
            raise ValueError(
                f"PDFParser: PDF declares {declared_pages} pages, exceeds cap of "
                f"{self.max_pages} (path={file_path})"
            )
        return reader

    def read(self, file_path: str) -> str:
        try:
            reader = self._open_reader(file_path)
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
            return "\n".join(parts).strip()
        except (ImportError, ValueError):
            raise
        except Exception as exc:
            logger.error("PDFParser failed for '%s': %s", file_path, exc)
            raise

    def read_pages(self, file_path: str) -> list[tuple[int, str]]:
        try:
            reader = self._open_reader(file_path)
            pages: list[tuple[int, str]] = []
            for i, page in enumerate(reader.pages, start=1):
                pages.append((i, page.extract_text() or ""))
            return pages
        except (ImportError, ValueError):
            raise
        except Exception as exc:
            logger.error("PDFParser failed for '%s': %s", file_path, exc)
            raise


class JSONParser(ParserStrategy):
    """Pretty-print JSON contents."""

    def read(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("JSONParser failed for '%s': %s", file_path, exc)
            raise


class YAMLParser(ParserStrategy):
    """Pretty-print YAML contents (requires pyyaml)."""

    def read(self, file_path: str) -> str:
        try:
            import yaml
        except Exception as exc:
            raise ImportError("YAMLParser requires PyYAML: pip install pyyaml") from exc
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            logger.error("YAMLParser failed for '%s': %s", file_path, exc)
            raise


class HTMLParser(ParserStrategy):
    """Sanitize and return visible text; record per-category removal counts."""

    _last_sanitization_counts: Optional[dict[str, int]] = None

    def read(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            text, counts = _sanitize_html(raw)
            self._last_sanitization_counts = counts or None
            return text
        except Exception as exc:
            logger.error("HTMLParser failed for '%s': %s", file_path, exc)
            raise


class CSVParser(ParserStrategy):
    """Render CSV as newline-joined rows with a soft row cap."""

    def read(self, file_path: str) -> str:
        try:
            out, limit = [], 200
            with open(file_path, "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for i, row in enumerate(reader):
                    out.append(", ".join(row))
                    if i + 1 >= limit:
                        out.append("..[truncated rows]")
                        break
            return "\n".join(out)
        except Exception as exc:
            logger.error("CSVParser failed for '%s': %s", file_path, exc)
            raise


class MarkdownParser(ParserStrategy):
    _last_sanitization_counts: Optional[dict[str, int]] = None

    def read(self, file_path: str) -> str:
        try:
            import markdown
        except Exception as exc:
            raise ImportError("MarkdownParser requires markdown: pip install markdown") from exc
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                html = markdown.markdown(fh.read())
            text, counts = _sanitize_html(html)
            self._last_sanitization_counts = counts or None
            return text
        except Exception as exc:
            logger.error("MarkdownParser failed for '%s': %s", file_path, exc)
            raise


class ParquetParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("ParquetParser requires pandas: pip install pandas") from exc
        try:
            df = pd.read_parquet(file_path)
            preview = df.head(30).to_string(index=False)
            schema = ", ".join(f"{c}:{str(t)}" for c, t in zip(df.columns, df.dtypes))
            return f"[SCHEMA] {schema}\n\n{preview}"
        except Exception as exc:
            logger.error("ParquetParser failed for '%s': %s", file_path, exc)
            raise


# NOTE: PickleParser was removed for security (CVE-class issue).
# `pickle.load` on attacker-controlled bytes is arbitrary code execution.
# Do not reintroduce. If pickle-shaped data must be ingested, the caller is
# responsible for converting it to a safe representation (e.g. JSON) before
# handing the file to UMA.


# ----------------------------- Registry -----------------------------------


@dataclass
class FileContentParser:
    """Centralized dispatcher (ext/MIME -> ParserStrategy).

    pdf_max_pages is propagated to every PDFParser instance the dispatcher
    constructs. None means use PDFParser.DEFAULT_MAX_PAGES.
    """

    _by_ext: dict[str, ParserStrategy] = None
    _by_mime: dict[str, ParserStrategy] = None
    pdf_max_pages: Optional[int] = None

    def __post_init__(self) -> None:
        self._by_ext = {}
        self._by_mime = {}

        # PDF parsers share the configured page cap so any caller path
        # (extension dispatch or MIME dispatch) gets the same defense.
        _pdf = PDFParser(max_pages=self.pdf_max_pages)

        self.register_parser(".txt", TXTParser())
        self.register_parser(".md", MarkdownParser())
        self.register_parser(".markdown", MarkdownParser())
        self.register_parser(".json", JSONParser())
        self.register_parser(".yaml", YAMLParser())
        self.register_parser(".yml", YAMLParser())
        self.register_parser(".csv", CSVParser())
        self.register_parser(".html", HTMLParser())
        self.register_parser(".htm", HTMLParser())
        self.register_parser(".xhtml", HTMLParser())
        self.register_parser(".pdf", _pdf)
        self.register_parser(".parquet", ParquetParser())
        # ".pkl" intentionally NOT registered: pickle.load is RCE-by-design.

        self.register_mime("text/plain", TXTParser())
        self.register_mime("text/markdown", TXTParser())
        self.register_mime("application/json", JSONParser())
        self.register_mime("application/x-yaml", YAMLParser())
        self.register_mime("text/html", HTMLParser())
        self.register_mime("application/pdf", _pdf)
        # "application/octet-stream" intentionally NOT registered: it is the
        # default MIME for unknown binary content and previously dispatched to
        # PickleParser, which is RCE-by-design. Unknown binary uploads must
        # fail closed in mime_check.enforce_mime_consistency.

    def register_parser(self, ext: str, impl: ParserStrategy) -> None:
        key = (ext or "").strip().lower()
        if not key.startswith("."):
            logger.warning("register_parser: extension %r should start with '.'", key)
        self._by_ext[key] = impl

    def register_mime(self, mime: str, impl: ParserStrategy) -> None:
        key = (mime or "").strip().lower()
        self._by_mime[key] = impl

    def supported_ext(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_ext.keys()))

    def supported_mime(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_mime.keys()))

    def read(self, file_path: str, *, content_type: Optional[str] = None) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()

        impl = self._by_ext.get(ext)
        if impl:
            logger.debug(
                "FileContentParser: using ext=%s parser=%s for %s",
                ext,
                impl.__class__.__name__,
                file_path,
            )
            return impl.read(file_path)

        if content_type:
            ct = content_type.split(";")[0].strip().lower()
            impl = self._by_mime.get(ct)
            if impl:
                logger.debug(
                    "FileContentParser: using mime=%s parser=%s for %s",
                    ct,
                    impl.__class__.__name__,
                    file_path,
                )
                return impl.read(file_path)

        if ext == ".txt":
            logger.debug("FileContentParser: falling back to TXTParser for %s", file_path)
            return TXTParser().read(file_path)
        if ext in (".md", ".markdown"):
            logger.debug("FileContentParser: falling back to MarkdownParser for %s", file_path)
            return MarkdownParser().read(file_path)

        raise ValueError(
            f"No parser registered for extension '{ext or '<none>'}' "
            f"and content_type '{content_type or '<none>'}'. "
            f"Known: ext={self.supported_ext()}, mime={self.supported_mime()}"
        )

    def parse(self, file_path: str, *, content_type: Optional[str] = None) -> ParsedDocument:
        """
        Parse a local file into ParsedDocument. PDFs preserve per-page text.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()

        impl = self._by_ext.get(ext)
        if impl is None and content_type:
            ct = content_type.split(";")[0].strip().lower()
            impl = self._by_mime.get(ct)
        if impl is None:
            raise ValueError(
                f"No parser registered for extension '{ext or '<none>'}' "
                f"and content_type '{content_type or '<none>'}'. "
                f"Known: ext={self.supported_ext()}, mime={self.supported_mime()}"
            )

        source_hash = _hash_file(file_path)
        doc_id = f"doc_{source_hash[:24]}"
        pages = impl.read_pages(file_path)
        san_counts = getattr(impl, "_last_sanitization_counts", None)
        return ParsedDocument(
            doc_id=doc_id,
            source_path=file_path,
            source_hash=source_hash,
            pages=pages,
            extracted_at=datetime.now(timezone.utc),
            sanitization_counts=san_counts or None,
        )


# ----------------------------- Public API ----------------------------------


def parse_file(
    file_path: str,
    *,
    content_type: Optional[str] = None,
    pdf_max_pages: Optional[int] = None,
) -> ParsedDocument:
    """
    Parse a local file into ParsedDocument.

    - PDFs preserve per-page text.
    - Other formats return a single-page text payload.

    pdf_max_pages caps the page count a PDF can declare before any text
    extraction runs (H2 defense against page-count amplification). None
    means use PDFParser.DEFAULT_MAX_PAGES.
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("parse_file: file_path must be a non-empty string")
    parser = FileContentParser(pdf_max_pages=pdf_max_pages)
    return parser.parse(file_path, content_type=content_type)
