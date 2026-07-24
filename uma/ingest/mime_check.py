from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class ContentType:
    detected: str          # e.g. "application/pdf"
    is_binary: bool


@dataclass(frozen=True)
class MimeCheckResult:
    declared_extension: str
    detected_type: str
    consistent: bool


class MimeRejection(ValueError):
    """Raised when a file's byte-level content type is inconsistent with its extension."""

    def __init__(self, *, detected_type: str, declared_extension: str, file_path: str) -> None:
        self.detected_type = detected_type
        self.declared_extension = declared_extension
        self.file_path = file_path
        super().__init__("mime mismatch or executable files are not supported")


class FileSizeRejection(ValueError):
    """Raised when a file exceeds the configured size cap.

    Distinct from MimeRejection so callers can log and report the two failure
    modes separately. Both fail closed before any parser is invoked.
    """

    def __init__(self, *, file_path: str, size_bytes: int, max_bytes: int) -> None:
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"file exceeds size cap: {size_bytes} > {max_bytes} bytes (path={file_path})"
        )


# (magic_bytes, offset, content_type, is_binary)
_BYTE_SIGNATURES: Sequence[Tuple[bytes, int, str, bool]] = [
    (b"%PDF-",                0, "application/pdf",            False),
    (b"PK\x03\x04",          0, "application/zip",            True),
    (b"MZ",                   0, "application/x-dosexec",      True),   # PE / Windows EXE
    (b"\x7fELF",              0, "application/x-elf",          True),
    (b"\xfe\xed\xfa\xce",    0, "application/x-mach-binary",  True),   # Mach-O 32-bit BE
    (b"\xce\xfa\xed\xfe",    0, "application/x-mach-binary",  True),   # Mach-O 32-bit LE
    (b"\xff\xfe\xfa\xce",    0, "application/x-mach-binary",  True),   # Mach-O 64-bit BE
    (b"\xcf\xfa\xed\xfe",    0, "application/x-mach-binary",  True),   # Mach-O 64-bit LE
    (b"{\\rtf",               0, "application/rtf",            False),
    (b"\x1f\x8b",             0, "application/gzip",           True),
    (b"\x89PNG\r\n\x1a\n",   0, "image/png",                  True),
    (b"\xff\xd8\xff",         0, "image/jpeg",                 True),
    (b"GIF87a",               0, "image/gif",                  True),
    (b"GIF89a",               0, "image/gif",                  True),
]

# Extensions that are never acceptable regardless of detected type
_ALWAYS_REJECTED_TYPES = frozenset({
    "application/x-dosexec",
    "application/x-elf",
    "application/x-mach-binary",
})

# Extensions that are never acceptable regardless of detected content type.
# These represent file formats where parsing is, by design, unsafe on
# attacker-controlled input (e.g. pickle.load is arbitrary code execution).
# Reject early at the MIME gate so they never reach a parser strategy.
_ALWAYS_REJECTED_EXTS = frozenset({
    ".pkl",
    ".pickle",
})

# extension → set of acceptable detected content types
_EXT_ALLOWED_TYPES: dict[str, frozenset[str]] = {
    ".pdf":      frozenset({"application/pdf"}),
    ".zip":      frozenset({"application/zip"}),
    ".rtf":      frozenset({"application/rtf"}),
    ".gz":       frozenset({"application/gzip"}),
    ".png":      frozenset({"image/png"}),
    ".jpg":      frozenset({"image/jpeg"}),
    ".jpeg":     frozenset({"image/jpeg"}),
    ".gif":      frozenset({"image/gif"}),
    ".html":     frozenset({"text/html", "text/plain"}),
    ".htm":      frozenset({"text/html", "text/plain"}),
    ".xhtml":    frozenset({"text/html", "text/plain"}),
    ".txt":      frozenset({"text/plain"}),
    ".md":       frozenset({"text/plain"}),
    ".markdown": frozenset({"text/plain"}),
    ".json":     frozenset({"text/plain"}),
    ".yaml":     frozenset({"text/plain"}),
    ".yml":      frozenset({"text/plain"}),
    ".csv":      frozenset({"text/plain"}),
}

_HEADER_BYTES = 16


def detect_content_type(file_path: str) -> ContentType:
    """Read the first 16 bytes and match against known signatures."""
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(_HEADER_BYTES)
    except OSError:
        return ContentType(detected="application/octet-stream", is_binary=True)

    for magic, offset, ctype, is_bin in _BYTE_SIGNATURES:
        end = offset + len(magic)
        if len(header) >= end and header[offset:end] == magic:
            return ContentType(detected=ctype, is_binary=is_bin)

    # Heuristic: if header contains a null byte, treat as opaque binary
    if b"\x00" in header:
        return ContentType(detected="application/octet-stream", is_binary=True)

    return ContentType(detected="text/plain", is_binary=False)


def check_mime_consistency(
    declared_extension: str,
    detected_type: ContentType,
) -> MimeCheckResult:
    """Return a result describing whether the extension matches the detected type."""
    ext = declared_extension.lower()
    dt = detected_type.detected

    # Extensions that are unsafe to parse on attacker input (e.g. pickle).
    # Reject regardless of detected content type.
    if ext in _ALWAYS_REJECTED_EXTS:
        return MimeCheckResult(
            declared_extension=ext,
            detected_type=dt,
            consistent=False,
        )

    # Executables are always rejected, regardless of declared extension
    if dt in _ALWAYS_REJECTED_TYPES:
        return MimeCheckResult(
            declared_extension=ext,
            detected_type=dt,
            consistent=False,
        )

    allowed = _EXT_ALLOWED_TYPES.get(ext)
    if allowed is None:
        # Unknown extension: fail closed on opaque binary content.
        # Text-detected payloads are allowed through (TXT/MD/Markdown parser
        # fallbacks in parser.FileContentParser.read handle them), but we
        # refuse to dispatch unknown-extension binaries to any parser.
        if detected_type.is_binary or dt == "application/octet-stream":
            return MimeCheckResult(declared_extension=ext, detected_type=dt, consistent=False)
        return MimeCheckResult(declared_extension=ext, detected_type=dt, consistent=True)

    consistent = dt in allowed
    return MimeCheckResult(declared_extension=ext, detected_type=dt, consistent=consistent)


def enforce_file_size_limit(file_path: str, max_bytes: int) -> int:
    """
    Raise FileSizeRejection if the file at file_path exceeds max_bytes.

    Returns the actual size in bytes on success. Bounds memory and CPU
    consumed by every downstream parser regardless of the file format.
    This is the first defense against decompression bombs, oversized PDFs,
    and malformed inputs that amplify into parser memory blow-ups.

    A max_bytes of 0 or negative means "no limit" — useful for tests and
    explicit opt-out, but should not be the production default.
    """
    if max_bytes is None or max_bytes <= 0:
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    try:
        size = os.path.getsize(file_path)
    except OSError:
        # If we can't stat the file, fail closed — every parser path past
        # here needs to read it, so an unreadable file is not ingestable.
        raise FileSizeRejection(file_path=file_path, size_bytes=0, max_bytes=max_bytes)

    if size > max_bytes:
        raise FileSizeRejection(file_path=file_path, size_bytes=size, max_bytes=max_bytes)
    return size


def enforce_mime_consistency(file_path: str, *, max_bytes: int | None = None) -> ContentType:
    """
    Detect the file's content type and raise MimeRejection if it is inconsistent
    with the declared extension or is an unconditionally blocked type.

    When max_bytes is provided and positive, also raise FileSizeRejection if
    the file exceeds the cap. Size is checked BEFORE content-type detection
    so an oversized file never gets its first 16 bytes opened — fail as
    early as possible.
    """
    if max_bytes is not None and max_bytes > 0:
        enforce_file_size_limit(file_path, max_bytes)
    ext = os.path.splitext(file_path)[1].lower() or ""
    ct = detect_content_type(file_path)
    result = check_mime_consistency(ext, ct)
    if not result.consistent:
        raise MimeRejection(
            detected_type=ct.detected,
            declared_extension=ext,
            file_path=file_path,
        )
    return ct