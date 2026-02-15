# PyMangle: Python Datalog Engine for Agentic Graph RAG

**Date**: 2026-02-15
**Status**: Approved
**Project**: agentic-graph-rag + standalone pymangle package

---

## Overview

Python re-implementation of a Google Mangle subset — a Datalog extension with temporal reasoning, aggregation, structured data, and external predicates. Integrates with agentic-graph-rag as a unified reasoning layer for query routing, graph inference, and access control.

**Approach**: Standalone `pymangle` package (reusable across projects) + integration layer in `agentic-graph-rag`.

---

## Architecture

### PyMangle Core (~5000 lines)

```
pymangle/
├── __init__.py              # Public API: Program, query(), load()
├── parser.py                # Lark grammar → AST (~600 lines)
├── ast_nodes.py             # Dataclasses: Atom, Clause, Decl, Term, Interval (~400)
├── analysis.py              # Stratification (Kosaraju SCC) + type inference (~500)
├── engine.py                # Semi-naive bottom-up evaluation (~800)
├── topdown.py               # Top-down evaluation for deferred/merge (~300)
├── unifier.py               # Unification + substitution (~200)
├── factstore.py             # In-memory indexed storage (~400)
├── builtins.py              # Built-in predicates & functions (~500)
├── temporal.py              # Temporal facts + Allen's intervals (~500)
├── external.py              # External predicate protocol (~200)
├── types.py                 # Optional type system (~300)
├── grammar.lark             # Lark EBNF grammar (~150)
└── tests/                   # ~80 tests
```

### Processing Pipeline

```
Source Text → [parse] → AST → [analysis] → [engine] → [factstore]
                 ↑                ↑              ↑           ↑
              Lark         Stratification   Semi-naive    Indexed
              grammar      (Kosaraju SCC)   bottom-up     in-memory
              (.lark)      + type check     + delta       store
```

### Integration Layer (~1500 lines)

```
agentic-graph-rag/
└── agentic_graph_rag/
    └── reasoning/
        ├── __init__.py
        ├── neo4j_bridge.py      # Neo4j → Mangle facts (external predicates)
        ├── reasoning_engine.py  # Facade: load rules + query
        ├── rules/
        │   ├── routing.mg       # Query routing rules
        │   ├── graph.mg         # Graph inference rules
        │   └── access.mg        # Access control policies
        └── tests/
```

---

## Parser (Lark)

EBNF grammar covering:

```ebnf
program     : (clause | decl | fact | query)*
clause      : atom ":-" body (";" body)* "."
fact        : atom ("@" interval)? "."
query       : atom "?"
body        : premise ("," premise)* transform?
premise     : atom | neg_atom | comparison | temporal_lit
neg_atom    : "!" atom
transform   : "|>" ("do" fn_call ",")? "let" var "=" fn_call
atom        : predicate "(" term ("," term)* ")"
term        : variable | constant | fn_call | list | map | struct
interval    : "[" time_expr "," time_expr "]"
```

---

## Engine: Semi-Naive Bottom-Up

5-phase evaluation:

1. **Stratify rules** — Kosaraju SCC + topological sort
2. **For each stratum**:
   - **Initial round** — apply all rules to existing facts
   - **Delta iteration** — delta-rules use only NEW facts (avoid re-derivation)
   - **Merge** — new facts into main store
   - **Fixpoint** — stop when no new facts
3. **Post-fixpoint** — do-transforms (aggregation) as post-processing

Data complexity: PTIME (pure Datalog fragment). Fact limit (default 100K) for protection against infinite derivations with functions.

---

## FactStore: Indexed In-Memory

Hash-index by predicate + first argument:

```python
class IndexedFactStore:
    _by_predicate: dict[str, set[Atom]]
    _by_first_arg: dict[tuple[str, Constant], set[Atom]]

    def add(self, fact: Atom) -> bool       # Returns True if new
    def query(self, pattern: Atom) -> Iterator[Atom]
    def merge(self, delta: FactStore) -> FactStore
```

---

## External Predicates (Neo4j Bridge)

Pull facts from Neo4j on-demand with filter pushdown:

```python
class Neo4jExternalPredicate:
    def __init__(self, driver, cypher_template: str): ...
    def query(self, inputs: list[Constant], filters: list) -> Iterator[list]: ...
```

Engine passes known arguments as `$input` to Cypher — Neo4j filters server-side.

---

## Temporal Reasoning

### Temporal Facts

Facts annotated with validity intervals: `fact(X, Y)@[start, end]`

- Concrete timestamps (ISO 8601)
- Unbounded: `_` for positive/negative infinity
- Variables: `@[S, E]` binds start/end during evaluation

### Allen's 9 Interval Relations (built-in)

`:interval:before`, `:interval:after`, `:interval:meets`, `:interval:overlaps`,
`:interval:during`, `:interval:contains`, `:interval:starts`, `:interval:finishes`,
`:interval:equals`

### Temporal FactStore

- Indexes facts by predicate + atom hash with interval list
- Interval coalescing: merges overlapping/adjacent intervals
- Max 1000 intervals per atom (prevents non-termination)
- Point queries and range queries

---

## Use Case 1: Query Routing Rules (`routing.mg`)

Replaces regex patterns in `router.py` with declarative rules:

```prolog
keyword(/relation, "связь").
keyword(/relation, "between").
keyword(/temporal, "когда").
keyword(/global, "overview").

match(Query, Category) :- query_contains(Query, Word), keyword(Category, Word).

match_count(Query, Category, Count) :-
    match(Query, Category)
    |> do fn:group_by(Query, Category), let Count = fn:count().

best_category(Query, Category, Count) :-
    match_count(Query, Category, Count), !better_match(Query, Category).

tool_for(/simple, "vector_search").
tool_for(/relation, "cypher_traverse").
tool_for(/multi_hop, "cypher_traverse").
tool_for(/global, "full_document_read").
tool_for(/temporal, "temporal_query").

route_to(Query, Tool, 0.7) :- best_category(Query, Category, _), tool_for(Category, Tool).
route_to(Query, "vector_search", 0.3) :- !match(Query, _).
```

**Benefit**: Users can add keywords without changing Python code.

---

## Use Case 2: Graph Reasoning (`graph.mg`)

Declarative inference rules over PhraseNode/PassageNode graph:

```prolog
Decl phrase_node(Name, Type, PageRank) external.
Decl edge(Source, Relation, Target) external.
Decl mentioned_in(Entity, PassageId, Text) external.

reachable(X, Y, 1) :- edge(X, _, Y).
reachable(X, Z, D) :- reachable(X, Y, D1), edge(Y, _, Z),
    D = fn:plus(D1, 1), D < 5.

common_neighbor(A, B, Neighbor) :- edge(A, _, Neighbor), edge(B, _, Neighbor), A != B.

edge_count(Node, Count) :- edge(Node, _, _) |> do fn:group_by(Node), let Count = fn:count().
hub_node(Node, Count) :- edge_count(Node, Count), Count > 5.

evidence_strength(Entity, Count) :-
    evidence(Entity, _) |> do fn:group_by(Entity), let Count = fn:count().

relevance_score(Entity, Score) :-
    phrase_node(Entity, _, PR), evidence_strength(Entity, EvidCount),
    Score = fn:plus(fn:mult(PR, 100.0), fn:float_div(EvidCount, 1.0)).
```

**Benefit**: Composable rules vs monolithic Cypher queries.

---

## Use Case 3: Access Control (`access.mg`)

Role-based policies for RAG result filtering:

```prolog
role_inherits(/admin, /analyst).
role_inherits(/analyst, /viewer).
has_role(User, Role) :- user_role(User, Role).
has_role(User, Parent) :- user_role(User, Child), role_inherits(Child, Parent).

resource_type("financial_data", /sensitive).
permit(/viewer, /read, /public).
permit(/analyst, /read, /sensitive).
permit(/admin, /read, /pii).

deny(_, /write, /pii).
deny(User, Action, Resource) :- user_restriction(User, Resource).

allowed(User, Action, Resource) :-
    has_role(User, Role), resource_type(Resource, Type),
    permit(Role, Action, Type), !deny(User, Action, Resource).

visible_passage(User, PassageId, Text) :-
    mentioned_in(Entity, PassageId, Text),
    phrase_node(Entity, EntityType, _),
    allowed(User, /read, EntityType).
```

**Benefit**: Deny overrides permit via stratification. Role inheritance via recursive rules. RAG filtering at Mangle level, before LLM.

---

## Data Flow: End to End

```
User Query
    │
    ▼
1. ReasoningEngine.classify_query()     → routing.mg → RouterDecision
    │
    ▼
2. RetrievalAgent.execute_tool()        → VectorCypher → raw chunks
    │
    ▼
3. ReasoningEngine.infer_connections()   → graph.mg (Neo4j external) → enriched context
    │
    ▼
4. ReasoningEngine.check_access()        → access.mg → filtered passages
    │
    ▼
5. Generator.generate()                  → LLM answer from filtered + enriched context
```

---

## Integration with Existing Router

Backward-compatible: Mangle first, fallback to pattern-based:

```python
class QueryRouter:
    def classify(self, query: str) -> RouterDecision:
        if self.reasoning:
            decision = self.reasoning.classify_query(query)
            if decision.confidence >= 0.5:
                return decision
        return self._pattern_classify(query)  # existing logic
```

---

## Testing Strategy

### PyMangle Core (~80 tests)

| Module | Tests | Focus |
|--------|-------|-------|
| parser | 15 | Facts, rules, negation, temporal, transforms, errors |
| unifier | 8 | Variable binding, occurs check, nested terms |
| engine | 20 | Fixpoint, delta iteration, stratification, limits |
| temporal | 12 | Intervals, Allen's relations, coalescing, point queries |
| builtins | 10 | Arithmetic, string, list, map, struct |
| types | 5 | Bounds checking, inclusion constraints |
| external | 5 | External predicate protocol, filter pushdown |
| integration | 5 | Full programs end-to-end |

### Integration Layer (~20 tests)

| Module | Tests | Focus |
|--------|-------|-------|
| routing | 8 | Keyword matching, fallback, confidence |
| graph | 5 | Reachable, hub, cluster (mock Neo4j) |
| access | 5 | Permit, deny, inheritance, filtering |
| bridge | 2 | Neo4j external predicate, pushdown |

### Coverage Target: 80%+, ~100 tests total

---

## Implementation Roadmap (10 Phases)

| # | Phase | Scope | Tests | ~Lines |
|---|-------|-------|-------|--------|
| 1 | AST + Parser | Lark grammar, AST dataclasses, parse() | 15 | 1000 |
| 2 | Unifier + FactStore | Substitution-based unification, indexed store | 8 | 600 |
| 3 | Engine Core | Semi-naive bottom-up, fixpoint, fact limit | 12 | 800 |
| 4 | Stratification + Negation | Kosaraju SCC, topo sort, negated premises | 8 | 500 |
| 5 | Builtins | Arithmetic, string, list, map, struct, comparison | 10 | 500 |
| 6 | Aggregation | Transform pipeline: group_by, count, sum, collect | 5 | 400 |
| 7 | Temporal | Temporal facts, interval store, Allen's relations | 12 | 500 |
| 8 | Type System + External | Optional bounds, external predicate protocol | 10 | 500 |
| 9 | Integration Layer | Neo4j bridge, ReasoningEngine, .mg rule files | 15 | 700 |
| 10 | Wiring + Benchmark | Router integration, Streamlit tab, benchmark | 5 | 500 |
| | **Total** | | **~100** | **~6000** |

### Phase Dependencies

```
Phase 1 (AST + Parser)
  ├── Phase 2 (Unifier + FactStore)
  │     └── Phase 3 (Engine Core)
  │           ├── Phase 4 (Stratification + Negation)  ─┐
  │           ├── Phase 5 (Builtins)                    │ parallel
  │           │     └── Phase 6 (Aggregation)           │
  │           └── Phase 7 (Temporal)                   ─┘
  │
  └── Phase 8 (Types + External) ← needs Phase 3
        └── Phase 9 (Integration) ← needs Phases 4-8
              └── Phase 10 (Wiring) ← needs Phase 9
```

---

## Deliverables

1. **`pymangle`** — standalone Python Datalog engine (~5K lines, ~80 tests)
2. **`agentic_graph_rag/reasoning/`** — integration layer (~1.5K lines, ~20 tests)
3. **Streamlit tab** — "Reasoning" (7th tab): rule editing, query testing, stratification viz
4. **Benchmark** — `agent_mangle` mode vs existing `agent_pattern` / `agent_llm`

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parser | Lark (EBNF) | Clean transformers, actively maintained, no codegen |
| Unification | Substitution-based | Simpler than union-find for Python; sufficient for our scale |
| Evaluation | Semi-naive bottom-up | Standard for production Datalog; delta optimization critical |
| Stratification | Kosaraju SCC | Textbook algorithm, handles negation + aggregation |
| Temporal store | Dict + interval list | Simpler than interval tree; coalescing prevents explosion |
| Fact limit | 100K default | Protects against infinite derivations with functions |
| Neo4j integration | External predicates | On-demand pull with filter pushdown; no bulk loading |
| Router integration | Fallback pattern | Backward compatible; Mangle first, regex fallback |
