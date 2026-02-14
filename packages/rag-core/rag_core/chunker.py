"""Semantic text chunker with markdown-aware splitting.

Merged from RAG 2.0 (chunk_text → list[Chunk]) and TKB (table-aware
chunking, sanitize_for_graphiti, split_large_content for KG episodes).
"""

from __future__ import annotations

import hashlib
import re

from rag_core.config import get_settings
from rag_core.models import Chunk

# ── KG episode utilities (from TKB) ─────────────────────────────

MAX_EPISODE_CHARS = 8_000

_LUCENE_SPECIAL_RE = re.compile(r'[+\-&|!(){}[\]^"~*?:\\/]')


def sanitize_for_graphiti(text: str) -> str:
    """Remove Lucene special characters that break Neo4j fulltext queries."""
    return _LUCENE_SPECIAL_RE.sub(" ", text)


def split_large_content(
    text: str,
    source: str,
    max_chars: int = MAX_EPISODE_CHARS,
) -> list[tuple[str, str]]:
    """Split large text into episode-sized pieces for Graphiti.

    Returns list of (content, source_name) tuples.
    """
    if len(text) <= max_chars:
        return [(text, source)]

    paragraphs = re.split(r"\n\s*\n", text)
    parts: list[tuple[str, str]] = []
    current = ""
    part_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            if current:
                parts.append((current.strip(), f"{source}_part_{part_num}"))
                part_num += 1
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > max_chars:
                    if current:
                        parts.append((current.strip(), f"{source}_part_{part_num}"))
                        part_num += 1
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current:
                parts.append((current.strip(), f"{source}_part_{part_num}"))
                part_num += 1
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        parts.append((current.strip(), f"{source}_part_{part_num}"))

    return parts


# ── Main chunk_text function (from RAG 2.0) ─────────────────────

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk text semantically using markdown structure.

    Strategy:
    1. Split by markdown headers (##, ###) first
    2. Then by paragraphs
    3. If still too large, split by sentences
    4. Tables (lines starting with |) kept as atomic units

    Each chunk gets auto-generated id (md5) and metadata.
    """
    cfg = get_settings()
    if chunk_size is None:
        chunk_size = cfg.indexing.chunk_size
    if chunk_overlap is None:
        chunk_overlap = cfg.indexing.chunk_overlap

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    sections = _split_by_headers(text)

    for section_title, section_content in sections:
        section_chunks = _chunk_section(section_content, chunk_size, chunk_overlap, section_title)
        chunks.extend(section_chunks)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


# ── Internal helpers ─────────────────────────────────────────────

def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split text by markdown headers (## or ###)."""
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_content: list[str] = []

    for line in text.split("\n"):
        match = header_pattern.match(line)
        if match:
            if current_content:
                sections.append((current_title, "\n".join(current_content)))
            current_title = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_title, "\n".join(current_content)))

    if not sections:
        sections.append(("", text))

    return sections


def _chunk_section(
    text: str, chunk_size: int, chunk_overlap: int, section_title: str,
) -> list[Chunk]:
    """Chunk a single section into Chunk objects."""
    if not text.strip():
        return []

    lines = text.split("\n")
    is_table = all(line.strip().startswith("|") or not line.strip() for line in lines)

    if is_table and text.strip():
        return [_create_chunk(text, section_title)]

    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_chunk_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk_text) + len(para) + 2 <= chunk_size:
            current_chunk_text = f"{current_chunk_text}\n\n{para}".strip() if current_chunk_text else para
        else:
            if current_chunk_text:
                chunks.append(_create_chunk(current_chunk_text, section_title))
                if chunk_overlap > 0:
                    current_chunk_text = current_chunk_text[-chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk_text = para
            else:
                sentence_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
                for sc in sentence_chunks:
                    chunks.append(_create_chunk(sc, section_title))
                current_chunk_text = ""

    if current_chunk_text:
        chunks.append(_create_chunk(current_chunk_text, section_title))

    return chunks


def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text by sentence boundaries when paragraph is too large."""
    sentence_pattern = re.compile(r"([.!?]+\s+)")
    parts = sentence_pattern.split(text)

    sentences: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    chunks: list[str] = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
                if chunk_overlap > 0:
                    current_chunk = current_chunk[-chunk_overlap:] + sent
                else:
                    current_chunk = sent
            else:
                chunks.append(sent)
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _create_chunk(content: str, section_title: str) -> Chunk:
    """Create Chunk with auto-generated id and metadata."""
    chunk_id = hashlib.md5(content.encode()).hexdigest()[:8]
    return Chunk(
        id=chunk_id,
        content=content,
        metadata={"section_title": section_title} if section_title else {},
    )
