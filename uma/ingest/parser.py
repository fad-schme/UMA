from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .types import ParsedDocument

logger = logging.getLogger(__name__)


def _hash_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ----------------------------- Strategy base -----------------------------


class ParserStrategy:
    """Synchronous parser returning a text string."""

    def read(self, file_path: str) -> str:
        raise NotImplementedError

    def read_pages(self, file_path: str) -> List[Tuple[int, str]]:
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
    def read(self, file_path: str) -> str:
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            raise ImportError("PDFParser requires PyPDF2: pip install PyPDF2") from exc
        try:
            reader = PdfReader(file_path)
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
            return "\n".join(parts).strip()
        except Exception as exc:
            logger.error("PDFParser failed for '%s': %s", file_path, exc)
            raise

    def read_pages(self, file_path: str) -> List[Tuple[int, str]]:
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            raise ImportError("PDFParser requires PyPDF2: pip install PyPDF2") from exc
        try:
            reader = PdfReader(file_path)
            pages: List[Tuple[int, str]] = []
            for i, page in enumerate(reader.pages, start=1):
                pages.append((i, page.extract_text() or ""))
            return pages
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
    """Strip scripts/styles and return visible text."""

    def read(self, file_path: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except Exception as exc:
            raise ImportError(
                "HTMLParser requires beautifulsoup4: pip install beautifulsoup4"
            ) from exc
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                soup = BeautifulSoup(fh, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            return soup.get_text(separator="\n", strip=True)
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
    def read(self, file_path: str) -> str:
        try:
            import markdown
        except Exception as exc:
            raise ImportError("MarkdownParser requires markdown: pip install markdown") from exc
        try:
            from bs4 import BeautifulSoup
        except Exception as exc:
            raise ImportError(
                "MarkdownParser requires beautifulsoup4: pip install beautifulsoup4"
            ) from exc
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                html = markdown.markdown(fh.read())
            text = "".join(BeautifulSoup(html, "html.parser").find_all(string=True))
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


class PickleParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        try:
            import pickle
        except Exception as exc:
            raise ImportError("PickleParser requires pickle (std lib).") from exc

        try:
            with open(file_path, "rb") as fh:
                data = pickle.load(fh)
            try:
                return json.dumps(data, ensure_ascii=False, default=str)
            except Exception:
                return repr(data)
        except Exception as exc:
            logger.error("PickleParser failed for '%s': %s", file_path, exc)
            raise


# ----------------------------- Registry -----------------------------------


@dataclass
class FileContentParser:
    """Centralized dispatcher (ext/MIME -> ParserStrategy)."""

    _by_ext: Dict[str, ParserStrategy] = None
    _by_mime: Dict[str, ParserStrategy] = None

    def __post_init__(self) -> None:
        self._by_ext = {}
        self._by_mime = {}

        self.register_parser(".txt", TXTParser())
        self.register_parser(".md", TXTParser())
        self.register_parser(".markdown", TXTParser())
        self.register_parser(".json", JSONParser())
        self.register_parser(".yaml", YAMLParser())
        self.register_parser(".yml", YAMLParser())
        self.register_parser(".csv", CSVParser())
        self.register_parser(".html", HTMLParser())
        self.register_parser(".htm", HTMLParser())
        self.register_parser(".xhtml", HTMLParser())
        self.register_parser(".pdf", PDFParser())
        self.register_parser(".parquet", ParquetParser())
        self.register_parser(".pkl", PickleParser())

        self.register_mime("text/plain", TXTParser())
        self.register_mime("text/markdown", TXTParser())
        self.register_mime("application/json", JSONParser())
        self.register_mime("application/x-yaml", YAMLParser())
        self.register_mime("text/html", HTMLParser())
        self.register_mime("application/pdf", PDFParser())
        self.register_mime("application/octet-stream", PickleParser())

    def register_parser(self, ext: str, impl: ParserStrategy) -> None:
        key = (ext or "").strip().lower()
        if not key.startswith("."):
            logger.warning("register_parser: extension %r should start with '.'", key)
        self._by_ext[key] = impl

    def register_mime(self, mime: str, impl: ParserStrategy) -> None:
        key = (mime or "").strip().lower()
        self._by_mime[key] = impl

    def supported_ext(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_ext.keys()))

    def supported_mime(self) -> Tuple[str, ...]:
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

        if ext in (".txt", ".md", ".markdown"):
            logger.debug("FileContentParser: falling back to TXTParser for %s", file_path)
            return TXTParser().read(file_path)

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
        return ParsedDocument(
            doc_id=doc_id,
            source_path=file_path,
            source_hash=source_hash,
            pages=pages,
            extracted_at=datetime.now(timezone.utc),
        )


# ----------------------------- Public API ----------------------------------


def parse_file(file_path: str, *, content_type: Optional[str] = None) -> ParsedDocument:
    """
    Parse a local file into ParsedDocument.

    - PDFs preserve per-page text.
    - Other formats return a single-page text payload.
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("parse_file: file_path must be a non-empty string")
    parser = FileContentParser()
    return parser.parse(file_path, content_type=content_type)
