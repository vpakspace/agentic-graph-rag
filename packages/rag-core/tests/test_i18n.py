"""Tests for rag_core.i18n."""

from rag_core.i18n import TRANSLATIONS, get_translator


class TestTranslations:
    def test_all_keys_have_en_and_ru(self):
        for key, entry in TRANSLATIONS.items():
            assert "en" in entry, f"Key '{key}' missing 'en'"
            assert "ru" in entry, f"Key '{key}' missing 'ru'"

    def test_app_title(self):
        assert TRANSLATIONS["app_title"]["en"] == "Agentic Graph RAG"

    def test_has_tab_keys(self):
        tab_keys = ["tab_ingest", "tab_search", "tab_graph", "tab_benchmark", "tab_settings"]
        for key in tab_keys:
            assert key in TRANSLATIONS

    def test_has_search_mode_keys(self):
        modes = ["search_mode_vector", "search_mode_graph", "search_mode_hybrid", "search_mode_agent"]
        for key in modes:
            assert key in TRANSLATIONS


class TestGetTranslator:
    def test_english_translation(self):
        t = get_translator("en")
        assert t("app_title") == "Agentic Graph RAG"

    def test_russian_translation(self):
        t = get_translator("ru")
        assert t("tab_ingest") == "Загрузка"
        assert t("search_button") == "Спросить"

    def test_missing_key_returns_key(self):
        t = get_translator("en")
        assert t("nonexistent_key") == "nonexistent_key"

    def test_format_substitution(self):
        t = get_translator("en")
        result = t("ingest_success", chunks=10, total=50)
        assert "10" in result
        assert "50" in result

    def test_format_substitution_ru(self):
        t = get_translator("ru")
        result = t("ingest_success", chunks=23, total=100)
        assert "23" in result
        assert "100" in result

    def test_format_with_float(self):
        t = get_translator("en")
        result = t("search_source_score", score=0.876)
        assert "0.876" in result

    def test_format_with_percent(self):
        t = get_translator("en")
        result = t("bench_accuracy", correct=8, total=10, pct=0.8)
        assert "8" in result
        assert "10" in result

    def test_missing_format_key_returns_template(self):
        t = get_translator("en")
        # Pass wrong kwargs — should return template without crashing
        result = t("ingest_success", wrong_key="value")
        assert "{chunks}" in result

    def test_unknown_language_falls_back_to_en(self):
        t = get_translator("fr")
        assert t("app_title") == "Agentic Graph RAG"

    def test_default_language_is_en(self):
        t = get_translator()
        assert t("app_title") == "Agentic Graph RAG"

    def test_error_key_with_msg(self):
        t = get_translator("en")
        result = t("error", msg="something went wrong")
        assert "something went wrong" in result

    def test_error_key_ru(self):
        t = get_translator("ru")
        result = t("error", msg="ошибка")
        assert "ошибка" in result
