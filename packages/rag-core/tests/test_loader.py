"""Tests for rag_core.loader."""

import pytest
from rag_core.loader import SUPPORTED_EXTENSIONS, DoclingLoader, DocumentResult, load_file


class TestDocumentResult:
    def test_defaults(self):
        r = DocumentResult(markdown="hello")
        assert r.markdown == "hello"
        assert r.tables == []
        assert r.images == []
        assert r.metadata == {}


class TestDoclingLoader:
    def test_init_defaults(self):
        loader = DoclingLoader()
        assert loader._use_gpu is False
        assert loader._converter is None

    def test_init_with_gpu(self):
        loader = DoclingLoader(use_gpu=True)
        assert loader._use_gpu is True

    def test_load_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        loader = DoclingLoader()
        result = loader.load(str(f))
        assert result.markdown == "Hello, world!"
        assert result.metadata["format"] == ".txt"
        assert result.metadata["pages"] == 1

    def test_load_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nParagraph", encoding="utf-8")
        loader = DoclingLoader()
        result = loader.load(str(f))
        assert "# Title" in result.markdown

    def test_load_file_not_found(self):
        loader = DoclingLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")

    def test_load_unsupported_format(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        loader = DoclingLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(str(f))

    def test_load_bytes_txt(self):
        loader = DoclingLoader()
        result = loader.load_bytes(b"raw text data", "doc.txt")
        assert result.markdown == "raw text data"

    def test_load_bytes_unsupported(self):
        loader = DoclingLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_bytes(b"data", "file.abc")


class TestLoadFile:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "simple.txt"
        f.write_text("simple content", encoding="utf-8")
        text = load_file(str(f))
        assert text == "simple content"

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_file("/no/such/file.txt")


class TestSupportedExtensions:
    def test_contains_pdf(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_contains_docx(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_contains_txt(self):
        assert ".txt" in SUPPORTED_EXTENSIONS
