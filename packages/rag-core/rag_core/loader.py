"""Document loader based on IBM Docling.

Merged from TKB (DoclingLoader class, table/image extraction) and
RAG 2.0 (GPU acceleration, simple load_file API).

Supports: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"}


@dataclass
class DocumentResult:
    """Result of document processing via Docling."""

    markdown: str
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DoclingLoader:
    """Document loader with lazy Docling initialization and GPU support.

    Uses lazy initialization — Docling models (~1-2 GB) are loaded on first call.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self._converter: DocumentConverter | None = None
        self._use_gpu = use_gpu

    def _get_converter(self) -> DocumentConverter:
        """Lazy-initialize Docling converter."""
        if self._converter is None:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True

            if self._use_gpu:
                try:
                    from docling.datamodel.accelerator_options import (
                        AcceleratorDevice,
                        AcceleratorOptions,
                    )

                    pipeline_options.accelerator_options = AcceleratorOptions(
                        device=AcceleratorDevice.AUTO
                    )
                    logger.info("GPU acceleration enabled for PDF")
                except ImportError:
                    logger.warning("GPU acceleration imports failed, falling back to CPU")

            from docling.datamodel.base_models import InputFormat

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        return self._converter

    def load(self, file_path: str | Path) -> DocumentResult:
        """Load a document and extract content with tables and images."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Plain text files — no Docling needed
        if path.suffix.lower() in {".txt", ".md"}:
            return DocumentResult(
                markdown=path.read_text(encoding="utf-8"),
                metadata={"format": path.suffix, "pages": 1},
            )

        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document

        tables = self._extract_tables(doc)
        images = self._extract_images(doc)
        markdown = doc.export_to_markdown()

        pages = getattr(doc, "num_pages", None)
        if callable(pages):
            pages = pages()

        metadata = {
            "format": path.suffix,
            "pages": pages,
            "tables_count": len(tables),
            "images_count": len(images),
        }

        logger.info("Loaded %d chars from %s (%d tables, %d images)",
                     len(markdown), path.name, len(tables), len(images))

        return DocumentResult(markdown=markdown, tables=tables, images=images, metadata=metadata)

    def load_bytes(self, data: bytes, filename: str) -> DocumentResult:
        """Load a document from bytes (for file upload handlers)."""
        import os
        import tempfile

        suffix = Path(filename).suffix
        if suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {suffix}")

        if suffix.lower() in {".txt", ".md"}:
            return DocumentResult(
                markdown=data.decode("utf-8", errors="replace"),
                metadata={"format": suffix, "pages": 1},
            )

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        path = Path(tmp_path)
        try:
            os.close(fd)
            path.write_bytes(data)
            return self.load(path)
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _extract_tables(doc: object) -> list[dict]:
        tables = []
        for item, _level in doc.iterate_items():  # type: ignore[attr-defined]
            if hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    page_num = None
                    if hasattr(item, "prov") and item.prov:
                        page_num = getattr(item.prov[0], "page_no", None)
                    tables.append({
                        "caption": getattr(item, "caption", "") or "",
                        "markdown": df.to_markdown(index=False),
                        "csv": df.to_csv(index=False),
                        "page": page_num,
                    })
                except Exception:
                    pass
        return tables

    @staticmethod
    def _extract_images(doc: object) -> list[dict]:
        images = []
        for item, _level in doc.iterate_items():  # type: ignore[attr-defined]
            if hasattr(item, "get_image"):
                try:
                    img = item.get_image(doc)
                    if img:
                        page_num = None
                        if hasattr(item, "prov") and item.prov:
                            page_num = getattr(item.prov[0], "page_no", None)
                        images.append({
                            "caption": getattr(item, "caption", "") or "",
                            "page": page_num,
                        })
                except Exception:
                    pass
        return images


# ── Convenience function (RAG 2.0 API) ──────────────────────────

def load_file(file_path: str, use_gpu: bool = False) -> str:
    """Load document and return markdown text.

    Simple API for pipelines that only need the text.
    """
    loader = DoclingLoader(use_gpu=use_gpu)
    result = loader.load(file_path)
    return result.markdown
