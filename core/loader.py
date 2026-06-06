"""
Uploaded-document loader: turn a visitor's file into searchable text.

This module exists for the "upload your own document" demo. A visitor hands us
a file (PDF, Word, or plain text); we extract its text so the same chunking +
retrieval engine can index it for that one session. The file is never saved to
disk or added to the permanent corpus, it lives only in memory for the
visitor's session, which is the safe, sandboxed design.

Design notes worth understanding:
- We support a SHORT allow-list of formats (pdf, docx, txt, md), not "anything".
  Accepting arbitrary file types from the public is an abuse surface; a tight
  allow-list is the responsible choice and is itself a security-minded signal.
- We enforce a size cap and a character cap. Without caps, one huge upload
  could exhaust memory or run up a large API/embedding bill. Bounding inputs
  is basic defensive engineering.
- Each loader is wrapped so a corrupt or weird file produces a clear error
  message, not a crash.

PDF and Word parsing use small, well-known libraries (pypdf, python-docx).
They are pure-Python and deploy cleanly on Streamlit Cloud.
"""

from __future__ import annotations
from dataclasses import dataclass
import io

# Caps: tune here. These protect memory and cost.
MAX_FILE_BYTES = 5 * 1024 * 1024      # 5 MB upload limit
MAX_CHARS = 120_000                    # ~ a few dozen pages of text

ALLOWED_EXTENSIONS = ("pdf", "docx", "txt", "md")


@dataclass
class LoadedDoc:
    name: str          # original filename, used as the citation source label
    text: str          # extracted plain text
    note: str = ""     # any warning to surface to the user (e.g. truncated)


def _extract_pdf(data: bytes) -> str:
    """Pull text out of a PDF, page by page. Scanned/image-only PDFs have no
    embedded text, so this can return little or nothing; we detect that case
    upstream and tell the user rather than silently returning empty."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # One bad page should not sink the whole document.
            continue
    return "\n\n".join(parts)


def _extract_docx(data: bytes) -> str:
    """Pull text from a Word .docx by reading its paragraphs."""
    import docx  # python-docx
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def _extract_txt(data: bytes) -> str:
    """Plain text or markdown: decode, tolerating odd bytes."""
    return data.decode("utf-8", errors="ignore")


def load_uploaded(filename: str, data: bytes) -> LoadedDoc:
    """Validate and extract text from one uploaded file.

    Raises ValueError with a friendly message on anything we will not accept,
    so the UI can show the reason cleanly.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Please upload a PDF, Word (.docx), "
            f"or text (.txt/.md) file."
        )
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(
            f"That file is {len(data)//(1024*1024)} MB, which is over the "
            f"{MAX_FILE_BYTES//(1024*1024)} MB demo limit. Try a smaller document."
        )

    try:
        if ext == "pdf":
            text = _extract_pdf(data)
        elif ext == "docx":
            text = _extract_docx(data)
        else:
            text = _extract_txt(data)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Could not read that file. It may be corrupted or password-protected. "
            f"({type(e).__name__})"
        )

    text = text.strip()
    if len(text) < 30:
        raise ValueError(
            "Could not find readable text in that file. If it is a scanned PDF "
            "(an image of text), it has no selectable text to search. Try a file "
            "with real text."
        )

    note = ""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        note = "The document was long, so only the first portion was indexed for this demo."

    return LoadedDoc(name=filename, text=text, note=note)
