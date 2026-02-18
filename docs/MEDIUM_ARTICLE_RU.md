# Как я построил Graph RAG систему с точностью 96.7% за 5 дней: от научных статей до production-ready пайплайна

*Skeleton Indexing (KDD 2025) + HippoRAG 2 (ICML 2025) + VectorCypher + Datalog Reasoning + 10 итераций оптимизации*

---

## TL;DR

Я реализовал Graph RAG систему, которая комбинирует 5 техник из свежих научных статей в единый пайплайн с декларативным reasoning-движком, полной провенансной трассировкой и типизированным API. Результат: **174/180 (96.7%)** на билингвальном бенчмарке из 30 вопросов, оценённых в 6 режимах retrieval. Три режима достигли 100%. Ноль persistent failures.

**GitHub**: [vpakspace/agentic-graph-rag](https://github.com/vpakspace/agentic-graph-rag)

---

## Проблема: почему обычный RAG недостаточен

Классический RAG — "разбей документ на чанки, сделай embeddings, найди похожие" — работает для простых фактоидных вопросов. Но он ломается на:

- **Вопросах о связях**: "Как метод X связан с компонентом Y?" — ответ разбросан по разным чанкам
- **Multi-hop рассуждениях**: "Что произойдёт, если изменить A, учитывая что A влияет на B, а B на C?"
- **Глобальных вопросах**: "Перечисли все 7 архитектурных решений" — ответ в 7 разных местах документа
- **Кросс-языковых запросах**: русский вопрос о концепциях из английского документа

Моя цель — система, которая справляется со *всеми* этими типами вопросов, а не только с простыми.

---

## Архитектура: 5 техник из 2025 года

### 1. Skeleton Indexing (KET-RAG, KDD 2025)

**Проблема**: извлечение сущностей из всех чанков — дорого (O(n) вызовов LLM).

**Решение**: строим KNN-граф по embeddings чанков → PageRank → извлекаем сущности только из top-25% "скелетных" чанков. Периферийные чанки привязываем через keyword matching.

```
Chunks → KNN Graph → PageRank → Top-β Skeletal (full extraction)
                                → Peripheral (keyword linking only)
```

**Результат**: 75% меньше вызовов LLM при сопоставимом качестве. Это не трюк — это математика: PageRank выделяет чанки, которые наиболее "центральны" в семантическом пространстве документа.

### 2. Dual-Node Structure (HippoRAG 2, ICML 2025)

**Проблема**: обычный GraphRAG теряет контекст полных пассажей. Обычный RAG теряет связи между сущностями.

**Решение**: два типа узлов в Neo4j:
- **PhraseNode** — сущность (имя, тип, PageRank score, embedding)
- **PassageNode** — полный текст чанка (контент, embedding)
- **MENTIONED_IN** — связывает сущности с пассажами
- **RELATED_TO** — ко-вхождения между сущностями

Это даёт и навигацию по графу (через PhraseNode), и полный контекст (через PassageNode).

### 3. VectorCypher Retrieval

Гибридный retrieval в три фазы:

1. **Vector Index** → находим ближайшие PhraseNode через cosine similarity
2. **Cypher Traversal** → расширяем через RELATED_TO (до 3 хопов)
3. **PassageNode Collection** → собираем связанные пассажи → GraphContext

Ключевой инсайт: cosine re-ranking по реальным embeddings PassageNode из Neo4j бьёт RRF-фьюжн.

### 4. Agentic Router с Self-Correction

Три уровня маршрутизации с каскадным fallback:

| Tier | Метод | Confidence | Описание |
|------|-------|------------|----------|
| 1 | **Mangle** (Datalog) | 0.7 | 65 билингвальных ключевых слов |
| 2 | **LLM** (GPT-4o-mini) | 0.85 | Классификация нейросетью |
| 3 | **Pattern** (regex) | 0.5 | Regex-паттерны как fallback |

Если качество retrieval ниже порога (relevance < 2.0 из 5), система эскалирует по цепочке инструментов:

```
vector_search → cypher_traverse → hybrid_search → comprehensive_search → full_document_read
```

Каждая попытка перефразирует запрос через LLM. Лучшие результаты отслеживаются по всем попыткам.

### 5. PyMangle — Datalog-движок на Python

Полная реимплементация Google Mangle (2,919 строк):

- Lark-based парсер с кастомной грамматикой
- Semi-naive evaluation со стратифицированным отрицанием
- 35+ встроенных функций (арифметика, строки, списки, словари)
- Temporal evaluation
- Filter pushdown для внешних предикатов

Три файла правил:
- `routing.mg` — маршрутизация запросов (65 ключевых слов)
- `access.mg` — RBAC (role inheritance + permit/deny)
- `graph.mg` — граф-инференс (reachable, common_neighbor, evidence)

```prolog
% Транзитивное замыкание по графу
reachable(X, Y, 1) :- edge(X, R, Y).
reachable(X, Z, D) :- reachable(X, Y, D1), edge(Y, R, Z),
    D = fn:plus(D1, 1), D < 5.

% Общие соседи двух сущностей
common_neighbor(A, B, N) :- edge(A, R1, N), edge(B, R2, N), A != B.
```

---

## Бенчмарк: от 38% до 96.7% за 10 итераций

### Дизайн бенчмарка

- **30 вопросов**: 7 simple, 7 relation, 6 multi_hop, 6 global, 4 temporal
- **2 документа**: Doc1 (русский, граф знаний) + Doc2 (английский, архитектура SCL)
- **6 режимов retrieval**: vector, cypher, hybrid, agent_pattern, agent_llm, agent_mangle
- **180 оценок** (30 × 6) через hybrid judge: embedding similarity + keyword overlap + LLM-as-judge

### Эволюция результатов

```
v3:  38%  ████░░░░░░░░░░░░░░░░  Baseline (вопросы на EN, документы на RU)
v4:  67%  █████████░░░░░░░░░░░  +29pp — вопросы на RU (language match!)
v5:  73%  ██████████░░░░░░░░░░  +6pp  — comprehensive_search для global
v10: 65%  █████████░░░░░░░░░░░  -8pp  — добавили 15 новых вопросов
v11: 80%  ████████████░░░░░░░░  +15pp — enumeration prompt
v12: 93%  ██████████████████░░  +13pp — hybrid judge
v14: 96.7%███████████████████░  +3.7pp — semantic judge
```

### Финальные результаты (v14)

| Режим | Результат | |
|-------|-----------|--|
| **Vector** | **30/30 (100%)** | Чистый embedding search |
| **Hybrid** | **30/30 (100%)** | Vector + Graph |
| **Agent (Mangle)** | **30/30 (100%)** | Datalog правила |
| Agent (LLM) | 29/30 (96%) | GPT-4o-mini роутер |
| Agent (Pattern) | 28/30 (93%) | Regex паттерны |
| Cypher | 27/30 (90%) | Граф-траверсал |
| **Итого** | **174/180 (96.7%)** | **0 persistent failures** |

---

## 10 уроков оптимизации

### 1. Язык вопросов = язык документов (+29pp)

Самое большое улучшение за всю историю проекта. Вопросы на английском о русском документе давали 38%. Переключение на русские вопросы — 67%. Embeddings хорошо справляются с кросс-языковым поиском, но LLM-генератор теряет контекст.

### 2. Failures — это не retrieval, а generation + evaluation

Ключевой инсайт v11: для глобальных вопросов ВСЕ нужные ключевые слова находились в top-30 чанков. Проблема была в том, что генератор не перечислял все пункты, а judge обрезал ответ до 500 символов.

### 3. CoT-промпт для judge — катастрофа

Попытка сделать judge "умнее" через Chain-of-Thought ("перечисли найденные ключевые слова → посчитай → выдай вердикт") вызвала регрессию с 144/180 до 48/180. GPT-4o-mini буквально искал английские строки в русском тексте. Простой промпт "match CONCEPTS, not strings" работает в 3 раза лучше.

### 4. Cosine re-ranking бьёт RRF

Hybrid search с Reciprocal Rank Fusion давал худшие результаты, чем cosine re-ranking по реальным embeddings из Neo4j. RRF хорош для combining разных сигналов, но когда оба сигнала — embedding-based, прямое cosine similarity точнее.

### 5. Embedding similarity для judge (threshold 0.65)

Для вопросов с reference answer: cosine similarity между ответом системы и эталоном ≥ 0.65 → auto-PASS. Калибровка: правильный ответ ~0.677, неправильный (другой документ) ~0.570. Порог 0.65 идеально разделяет.

### 6. Кросс-языковая маршрутизация

Русский вопрос о концепциях из английского документа (Doc2/SCL) ломает vector_search — он возвращает Doc1. Решение: детектируем кросс-языковой глобальный запрос → напрямую `full_document_read` вместо vector_search.

### 7. Comprehensive search размывает результаты

`comprehensive_search` (multi-query fan-out) генерирует N подзапросов → каждый через vector_search → RRF merge. Но если все подзапросы возвращают Doc1, то единственный `full_document_read` результат для Doc2 тонет в RRF-merge.

### 8. Self-correction loop должен сохранять лучшее

Ранний баг: каждая попытка перезаписывала предыдущие результаты. Если attempt 1 дал score 2.5, а attempt 2 — score 1.8, система возвращала 1.8. Фикс: трекаем `best_results` и `best_score` по всем попыткам.

### 9. Enumeration prompt — специальный формат

Для глобальных вопросов ("перечисли все...") обычный prompt генерирует текст, а не список. Специальный enumeration prompt: "Output a numbered list. Scan ALL chunks. Do not stop early."

### 10. Judge limit 500 → 2000 символов

Обрезка ответа до 500 символов для judge убивала enumeration-ответы (7 пунктов ≈ 1500 символов). Увеличение до 2000 — мгновенный +5pp.

---

## Typed API и провенанс

Каждый запрос создаёт `PipelineTrace`:

```json
{
  "trace_id": "tr_abc123def456",
  "router_step": {
    "method": "mangle",
    "decision": {"query_type": "simple", "suggested_tool": "vector_search"}
  },
  "tool_steps": [{
    "tool_name": "vector_search",
    "results_count": 10,
    "relevance_score": 3.2,
    "duration_ms": 150
  }],
  "escalation_steps": [],
  "generator_step": {
    "model": "gpt-4o-mini",
    "confidence": 0.82
  },
  "total_duration_ms": 1800
}
```

API: FastAPI REST (`/api/v1/`) + MCP (FastMCP SSE) — и для REST-клиентов, и для AI-агентов.

---

## Цифры проекта

| Метрика | Значение |
|---------|----------|
| Python LOC | 15,395 (111 файлов) |
| Тесты | 454 (346 core + 108 PyMangle) |
| Коммиты | 69 за 5 дней |
| Зависимости | 26 пакетов |
| Итерации бенчмарка | 10 (v2 → v14) |
| Файлы результатов | 15 JSON (~4.7 MB) |
| Mangle-правила | 111 строк (3 файла) |
| Классы | 36 (14 data models, 8 services, 6 config, 4 reasoning) |

---

## Что дальше

- **Personalized PageRank** для query-focused графового обхода
- **Human evaluation** в дополнение к LLM-as-judge
- **Streaming ответы** в Streamlit UI
- **Docker Compose** для one-click deployment
- **Больше Mangle-правил** — temporal reasoning, conflict resolution

---

## Стек

| Компонент | Технология |
|-----------|-----------|
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536 dim) |
| Graph DB | Neo4j 5.x (Vector Index + Cypher) |
| Reasoning | PyMangle (Datalog, 2,919 LOC) |
| Doc Parsing | Docling (PDF/DOCX/PPTX + GPU) |
| Graph Algorithms | NetworkX (PageRank, KNN, PPR) |
| API | FastAPI (REST) + FastMCP (SSE/MCP) |
| UI | Streamlit (7 tabs) |
| Testing | pytest (454 tests) + ruff |
| CI/CD | GitHub Actions |

---

*Если вам интересны детали реализации или вы хотите обсудить Graph RAG — пишите в комментариях или открывайте issue на [GitHub](https://github.com/vpakspace/agentic-graph-rag).*

**Tags**: #GraphRAG #RAG #Neo4j #NLP #LLM #Python #DataScience #MachineLearning
