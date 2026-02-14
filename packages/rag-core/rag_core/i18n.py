"""Internationalization (i18n) for Agentic Graph RAG UI.

Merged from RAG 2.0 + TKB — unified translation keys for Streamlit UI.
"""

from __future__ import annotations

from typing import Callable

TRANSLATIONS: dict[str, dict[str, str]] = {
    # App-level
    "app_title": {"en": "Agentic Graph RAG", "ru": "Agentic Graph RAG"},
    "app_subtitle": {
        "en": "Skeleton Indexing + VectorCypher + Agentic Router",
        "ru": "Skeleton Indexing + VectorCypher + Agentic Router",
    },
    "language": {"en": "Language", "ru": "Язык"},

    # Tabs
    "tab_ingest": {"en": "Ingest", "ru": "Загрузка"},
    "tab_search": {"en": "Search & Q&A", "ru": "Поиск и Q&A"},
    "tab_graph": {"en": "Knowledge Graph", "ru": "Граф знаний"},
    "tab_benchmark": {"en": "Benchmark", "ru": "Benchmark"},
    "tab_settings": {"en": "Settings & Stats", "ru": "Настройки"},

    # Ingest tab
    "ingest_header": {"en": "Document Ingestion", "ru": "Загрузка документов"},
    "ingest_upload": {"en": "Upload a document", "ru": "Загрузите документ"},
    "ingest_source_upload": {"en": "Upload file", "ru": "Загрузить файл"},
    "ingest_source_path": {"en": "File path", "ru": "Путь к файлу"},
    "ingest_path_input": {"en": "Enter full file path", "ru": "Введите полный путь к файлу"},
    "ingest_path_placeholder": {
        "en": "/mnt/c/Users/.../document.pdf or /home/.../file.txt",
        "ru": "/mnt/c/Users/.../document.pdf или /home/.../file.txt",
    },
    "ingest_path_not_found": {"en": "File not found: {path}", "ru": "Файл не найден: {path}"},
    "ingest_supported": {
        "en": "Supported formats: TXT, PDF, DOCX, PPTX, XLSX, HTML",
        "ru": "Поддерживаемые форматы: TXT, PDF, DOCX, PPTX, XLSX, HTML",
    },
    "ingest_skip_enrichment": {
        "en": "Skip enrichment (faster, lower quality)",
        "ru": "Пропустить обогащение (быстрее, ниже качество)",
    },
    "ingest_button": {"en": "Ingest Document", "ru": "Загрузить в базу"},
    "ingest_loading": {"en": "Loading document...", "ru": "Загрузка документа..."},
    "ingest_chunking": {"en": "Chunking text...", "ru": "Разбиение на чанки..."},
    "ingest_enriching": {
        "en": "Enriching chunks with contextual retrieval...",
        "ru": "Обогащение чанков контекстом...",
    },
    "ingest_embedding": {"en": "Generating embeddings...", "ru": "Генерация эмбеддингов..."},
    "ingest_storing": {"en": "Storing in Neo4j...", "ru": "Сохранение в Neo4j..."},
    "ingest_success": {
        "en": "Ingested {chunks} chunks. Total in store: {total}",
        "ru": "Загружено {chunks} чанков. Всего в базе: {total}",
    },
    "ingest_chars_loaded": {"en": "Loaded {chars} characters", "ru": "Загружено {chars} символов"},
    "ingest_chunks_created": {"en": "Created {count} chunks", "ru": "Создано {count} чанков"},
    "ingest_kg_episodes": {
        "en": "Ingested {count} KG episodes",
        "ru": "Загружено {count} эпизодов в KG",
    },
    "ingest_gpu": {"en": "GPU Acceleration", "ru": "GPU ускорение"},

    # Search tab
    "search_header": {"en": "Search & Question Answering", "ru": "Поиск и ответы на вопросы"},
    "search_input": {"en": "Enter your question", "ru": "Введите вопрос"},
    "search_placeholder": {"en": "What is ...?", "ru": "Что такое ...?"},
    "search_mode": {"en": "Search mode", "ru": "Режим поиска"},
    "search_mode_vector": {"en": "Vector only", "ru": "Только вектор"},
    "search_mode_graph": {"en": "Graph only", "ru": "Только граф"},
    "search_mode_hybrid": {"en": "Hybrid (Vector + Graph)", "ru": "Гибридный (Вектор + Граф)"},
    "search_mode_agent": {"en": "Agent (auto-route)", "ru": "Agent (автомаршрутизация)"},
    "search_button": {"en": "Ask", "ru": "Спросить"},
    "search_thinking": {"en": "Searching and generating answer...", "ru": "Поиск и генерация ответа..."},
    "search_answer": {"en": "Answer", "ru": "Ответ"},
    "search_confidence": {"en": "Confidence", "ru": "Уверенность"},
    "search_retries": {"en": "Retries: {count}", "ru": "Повторных попыток: {count}"},
    "search_sources": {"en": "Sources ({count})", "ru": "Источники ({count})"},
    "search_source_score": {"en": "Score: {score:.3f}", "ru": "Score: {score:.3f}"},
    "search_no_results": {
        "en": "No answer generated. Make sure documents are ingested.",
        "ru": "Ответ не сгенерирован. Убедитесь, что документы загружены.",
    },
    "search_query_type": {"en": "Query type: {qtype}", "ru": "Тип запроса: {qtype}"},
    "search_router_confidence": {
        "en": "Router confidence: {conf:.0%}",
        "ru": "Уверенность роутера: {conf:.0%}",
    },

    # Tabs (extended)
    "tab_agent_trace": {"en": "Agent Trace", "ru": "Трассировка"},
    "tab_graph_explorer": {"en": "Graph Explorer", "ru": "Граф"},

    # Graph Explorer tab
    "graph_header": {"en": "Graph Explorer", "ru": "Исследователь графа"},
    "graph_entities": {"en": "Entities ({count})", "ru": "Сущности ({count})"},
    "graph_relationships": {"en": "Relationships ({count})", "ru": "Связи ({count})"},
    "graph_temporal": {"en": "Temporal Facts", "ru": "Временные факты"},
    "graph_phrase_nodes": {"en": "Phrase Nodes", "ru": "Узлы-фразы"},
    "graph_passage_nodes": {"en": "Passage Nodes", "ru": "Узлы-пассажи"},
    "graph_no_data": {"en": "No graph data. Ingest documents first.", "ru": "Нет данных. Сначала загрузите документы."},
    "graph_max_nodes": {"en": "Max nodes to display", "ru": "Макс. узлов для отображения"},

    # Agent Trace tab
    "trace_header": {"en": "Agent Trace", "ru": "Трассировка агента"},
    "trace_no_data": {
        "en": "Run a query in Search tab to see the trace.",
        "ru": "Выполните запрос во вкладке Поиск, чтобы увидеть трассировку.",
    },
    "trace_routing": {"en": "Routing Decision", "ru": "Решение роутера"},
    "trace_query_type": {"en": "Query Type", "ru": "Тип запроса"},
    "trace_confidence": {"en": "Confidence", "ru": "Уверенность"},
    "trace_reasoning": {"en": "Reasoning", "ru": "Обоснование"},
    "trace_tool": {"en": "Selected Tool", "ru": "Выбранный инструмент"},
    "trace_correction": {"en": "Self-Correction Steps", "ru": "Шаги самокоррекции"},
    "trace_retries": {"en": "Retries: {count}", "ru": "Повторов: {count}"},
    "trace_tool_calls": {"en": "Tool Calls", "ru": "Вызовы инструментов"},

    # Benchmark tab
    "bench_header": {"en": "Benchmark Evaluation", "ru": "Оценка качества"},
    "bench_mode": {"en": "Benchmark mode", "ru": "Режим benchmark"},
    "bench_run": {"en": "Run Benchmark", "ru": "Запустить Benchmark"},
    "bench_running": {
        "en": "Running benchmark ({current}/{total})...",
        "ru": "Запуск benchmark ({current}/{total})...",
    },
    "bench_results": {"en": "Benchmark Results", "ru": "Результаты Benchmark"},
    "bench_accuracy": {
        "en": "Accuracy: {correct}/{total} ({pct:.0%})",
        "ru": "Точность: {correct}/{total} ({pct:.0%})",
    },
    "bench_avg_confidence": {
        "en": "Avg Confidence: {conf:.2f}",
        "ru": "Средняя уверенность: {conf:.2f}",
    },
    "bench_col_q": {"en": "Q#", "ru": "Q#"},
    "bench_col_status": {"en": "Status", "ru": "Статус"},
    "bench_col_confidence": {"en": "Confidence", "ru": "Уверенность"},
    "bench_col_retries": {"en": "Retries", "ru": "Повторы"},
    "bench_col_question": {"en": "Question", "ru": "Вопрос"},

    # Settings tab
    "settings_header": {"en": "Settings & Statistics", "ru": "Настройки и статистика"},
    "settings_current": {"en": "Current Configuration", "ru": "Текущая конфигурация"},
    "settings_store_stats": {"en": "Vector Store Statistics", "ru": "Статистика Vector Store"},
    "settings_total_chunks": {"en": "Total chunks: {count}", "ru": "Всего чанков: {count}"},
    "settings_kg_entities": {"en": "KG entities: {count}", "ru": "Сущностей в KG: {count}"},
    "settings_clear_db": {"en": "Clear Database", "ru": "Очистить базу данных"},
    "settings_clear_confirm": {"en": "Type DELETE to confirm", "ru": "Введите DELETE для подтверждения"},
    "settings_clear_button": {"en": "Clear All Data", "ru": "Удалить все данные"},
    "settings_cleared": {"en": "Deleted {count} chunks", "ru": "Удалено {count} чанков"},

    # Settings — cache stats
    "settings_cache_header": {"en": "Cache Statistics", "ru": "Статистика кеша"},
    "settings_cache_size": {"en": "Entries: {size}/{max_size}", "ru": "Записей: {size}/{max_size}"},
    "settings_cache_hit_rate": {"en": "Hit rate: {rate:.0%}", "ru": "Попадания: {rate:.0%}"},
    "settings_monitor_header": {"en": "Query Monitor", "ru": "Монитор запросов"},
    "settings_monitor_total": {"en": "Total queries: {count}", "ru": "Всего запросов: {count}"},
    "settings_suggestions": {"en": "Tuning Suggestions", "ru": "Рекомендации по настройке"},

    # Common
    "error": {"en": "Error: {msg}", "ru": "Ошибка: {msg}"},
}


def get_translator(lang: str = "en") -> Callable[..., str]:
    """Return a translator function t(key, **kwargs) for the given language.

    Usage:
        t = get_translator("ru")
        t("ingest_success", chunks=23, total=100)
    """

    def t(key: str, **kwargs: object) -> str:
        entry = TRANSLATIONS.get(key)
        if entry is None:
            return key
        text = entry.get(lang, entry.get("en", key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    return t
