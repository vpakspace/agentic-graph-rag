# PyMangle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python Datalog engine (Mangle subset) and integrate it into agentic-graph-rag as a reasoning layer for query routing, graph inference, and access control.

**Architecture:** Standalone `pymangle` package with Lark-based parser, semi-naive bottom-up engine, temporal reasoning, and external predicate protocol. Separate integration layer in `agentic_graph_rag/reasoning/` connects to Neo4j and existing router.

**Tech Stack:** Python 3.12, Lark (parser), pytest, Pydantic, neo4j-driver

**Design Doc:** `docs/plans/2026-02-15-pymangle-design.md`

**Conventions:**
- `from __future__ import annotations` first in every file
- Type hints with `|` unions and built-in generics (`list[str]`)
- Absolute imports only
- `logging.getLogger(__name__)` for logging
- Private constants: `_UPPERCASE`
- Pydantic models for data structures
- Tests: class-based (`class TestX:`), `MagicMock` for mocking
- Line length: 120, ruff: E/F/I/W

---

## Phase 1: AST + Parser

### Task 1.1: Project Scaffolding

**Files:**
- Create: `pymangle/pyproject.toml`
- Create: `pymangle/pymangle/__init__.py`
- Create: `pymangle/pymangle/ast_nodes.py`
- Create: `pymangle/tests/__init__.py`

**Step 1: Create pymangle directory**

```bash
mkdir -p ~/agentic-graph-rag/pymangle/pymangle
mkdir -p ~/agentic-graph-rag/pymangle/tests
```

**Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pymangle"
version = "0.1.0"
description = "Python Datalog engine — Mangle subset"
requires-python = ">=3.12"
dependencies = ["lark>=1.1.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "ruff>=0.1.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.setuptools.packages.find]
where = ["."]
include = ["pymangle*"]
```

**Step 3: Write AST dataclasses**

```python
# pymangle/pymangle/ast_nodes.py
"""AST nodes for Mangle/Datalog programs."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class TermType(enum.Enum):
    VARIABLE = "variable"
    CONSTANT = "constant"
    NAME = "name"           # /name_constant
    NUMBER = "number"
    FLOAT = "float"
    STRING = "string"
    FUNCALL = "funcall"
    LIST = "list"
    MAP = "map"
    STRUCT = "struct"


@dataclass(frozen=True)
class Variable:
    """Logic variable (uppercase start)."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant:
    """Ground value: number, string, or name."""
    value: str | int | float
    type: TermType = TermType.STRING

    def __repr__(self) -> str:
        if self.type == TermType.NAME:
            return f"/{self.value}"
        if self.type == TermType.STRING:
            return f'"{self.value}"'
        return str(self.value)


@dataclass(frozen=True)
class FunCall:
    """Function call: fn:name(args...)."""
    name: str
    args: tuple[Term, ...] = ()

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass(frozen=True)
class ListTerm:
    """List literal: fn:list(a, b, c)."""
    elements: tuple[Term, ...] = ()


@dataclass(frozen=True)
class MapTerm:
    """Map literal: fn:map(key1, val1, key2, val2)."""
    entries: tuple[tuple[Term, Term], ...] = ()


@dataclass(frozen=True)
class StructTerm:
    """Struct literal: fn:struct(/field, val, ...)."""
    fields: tuple[tuple[str, Term], ...] = ()


# Union type for all terms
Term = Variable | Constant | FunCall | ListTerm | MapTerm | StructTerm


@dataclass(frozen=True)
class PredicateSym:
    """Predicate symbol with arity."""
    name: str
    arity: int


@dataclass(frozen=True)
class Atom:
    """Predicate application: p(t1, t2, ...)."""
    predicate: str
    args: tuple[Term, ...]

    @property
    def arity(self) -> int:
        return len(self.args)

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.predicate}({args_str})"


@dataclass(frozen=True)
class NegAtom:
    """Negated atom: !p(X)."""
    atom: Atom


@dataclass(frozen=True)
class Comparison:
    """Comparison: X != Y, X < Y, etc."""
    left: Term
    op: str  # "==", "!=", "<", "<=", ">", ">="
    right: Term


@dataclass(frozen=True)
class Interval:
    """Time interval: [start, end]."""
    start: Term  # datetime constant, variable, or _ (unbounded)
    end: Term


@dataclass(frozen=True)
class TemporalAtom:
    """Atom with temporal annotation: p(X)@[S, E]."""
    atom: Atom
    interval: Interval


# Body premise types
Premise = Atom | NegAtom | Comparison | TemporalAtom


@dataclass(frozen=True)
class Transform:
    """Aggregation pipeline: |> do fn:group_by(...), let Var = fn:reducer()."""
    group_by: tuple[Term, ...] = ()
    variable: Variable | None = None
    reducer: FunCall | None = None


@dataclass(frozen=True)
class Clause:
    """Rule: head :- body."""
    head: Atom
    premises: tuple[Premise, ...] = ()
    transform: Transform | None = None
    head_interval: Interval | None = None  # temporal head


@dataclass(frozen=True)
class BoundDecl:
    """Type bound: bound [/type1, /type2]."""
    types: tuple[str, ...]


class DeclDescriptor(enum.Enum):
    EXTERNAL = "external"
    TEMPORAL = "temporal"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class Decl:
    """Declaration: Decl pred(X, Y) descriptor bound [types]."""
    predicate: str
    arity: int
    descriptors: tuple[DeclDescriptor, ...] = ()
    bounds: tuple[BoundDecl, ...] = ()


@dataclass
class Program:
    """Complete Mangle program."""
    clauses: list[Clause] = field(default_factory=list)
    facts: list[Clause] = field(default_factory=list)  # Rules with empty body
    decls: list[Decl] = field(default_factory=list)

    def all_clauses(self) -> list[Clause]:
        return self.facts + self.clauses
```

**Step 4: Write `__init__.py`**

```python
# pymangle/pymangle/__init__.py
"""PyMangle — Python Datalog engine (Mangle subset)."""
from __future__ import annotations

from pymangle.ast_nodes import (
    Atom,
    Clause,
    Comparison,
    Constant,
    Decl,
    FunCall,
    Interval,
    NegAtom,
    Program,
    TemporalAtom,
    Transform,
    Variable,
)

__version__ = "0.1.0"

__all__ = [
    "Atom",
    "Clause",
    "Comparison",
    "Constant",
    "Decl",
    "FunCall",
    "Interval",
    "NegAtom",
    "Program",
    "TemporalAtom",
    "Transform",
    "Variable",
]
```

**Step 5: Install in dev mode and verify**

```bash
cd ~/agentic-graph-rag && pip install -e pymangle --no-deps
python -c "import pymangle; print(pymangle.__version__)"
```
Expected: `0.1.0`

**Step 6: Commit**

```bash
git add pymangle/
git commit -m "feat(pymangle): project scaffolding + AST dataclasses"
```

---

### Task 1.2: Lark Grammar

**Files:**
- Create: `pymangle/pymangle/grammar.lark`
- Create: `pymangle/tests/test_grammar.py`

**Step 1: Write Lark grammar**

```lark
// pymangle/pymangle/grammar.lark
// Mangle/Datalog grammar for Lark parser

start: statement*

statement: clause | fact | decl

// Declarations
decl: "Decl" PREDICATE "(" decl_args ")" descriptor* bound* "."
decl_args: VARIABLE ("," VARIABLE)*
descriptor: "external" | "temporal" | "deferred"
bound: "bound" "[" type_list "]"
type_list: type_expr ("," type_expr)*
type_expr: NAME_CONST | "/" IDENT

// Facts (rules with empty body) + optional temporal
fact: atom temporal_annot? "."

// Rules
clause: atom temporal_annot? ":-" body "."
body: premise ("," premise)* transform?

// Premises
premise: neg_atom | comparison | temporal_atom | atom
neg_atom: "!" atom
temporal_atom: atom temporal_annot
comparison: term COMP_OP term
COMP_OP: "==" | "!=" | "<" | "<=" | ">" | ">="

// Temporal annotation
temporal_annot: "@" "[" time_expr "," time_expr "]"
time_expr: term | "_" | ISO_DATE | ISO_DATETIME

// Atoms
atom: PREDICATE "(" args? ")"
     | BUILTIN_PRED "(" args? ")"
args: term ("," term)*

// Transform (aggregation pipeline)
transform: "|>" do_clause? let_clause
do_clause: "do" funcall ","
let_clause: "let" VARIABLE "=" funcall

// Terms
?term: funcall
     | list_term
     | map_term
     | struct_term
     | variable
     | constant

variable: VARIABLE
constant: STRING | NUMBER | FLOAT | NAME_CONST

funcall: FN_NAME "(" args? ")"
list_term: "fn:list" "(" args? ")"
map_term: "fn:map" "(" args? ")"
struct_term: "fn:struct" "(" args? ")"

// Terminals
PREDICATE: /[a-z_][a-z0-9_]*/
BUILTIN_PRED: /:[a-z_]+(?::[a-z_]+)*/
VARIABLE: /[A-Z][A-Za-z0-9_]*/
FN_NAME: /fn:[a-z_]+(?::[a-z_]+)*/
NAME_CONST: "/" IDENT
IDENT: /[a-z_][a-z0-9_]*/
STRING: /\"[^\"]*\"/
NUMBER: /\-?[0-9]+/
FLOAT: /\-?[0-9]+\.[0-9]+/
ISO_DATE: /\d{4}-\d{2}-\d{2}/
ISO_DATETIME: /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/

%import common.WS
%import common.SH_COMMENT
%ignore WS
%ignore SH_COMMENT
// Also ignore Mangle-style comments
%ignore /%.*/
```

**Step 2: Write grammar smoke test**

```python
# pymangle/tests/test_grammar.py
"""Tests for Lark grammar parsing."""
from __future__ import annotations

from lark import Lark
from pathlib import Path

GRAMMAR_PATH = Path(__file__).parent.parent / "pymangle" / "grammar.lark"


def _parser() -> Lark:
    return Lark(GRAMMAR_PATH.read_text(), start="start", parser="earley")


class TestGrammarFacts:
    def test_simple_fact(self):
        tree = _parser().parse('edge("a", "b").')
        assert tree.data == "start"

    def test_name_constant_fact(self):
        tree = _parser().parse("tool_for(/simple, \"vector_search\").")
        assert tree.data == "start"

    def test_numeric_fact(self):
        tree = _parser().parse("score(42, 3.14).")
        assert tree.data == "start"

    def test_temporal_fact(self):
        tree = _parser().parse('link("a", "b")@[2024-01-01, 2024-12-31].')
        assert tree.data == "start"


class TestGrammarRules:
    def test_simple_rule(self):
        tree = _parser().parse("reachable(X, Z) :- edge(X, Y), edge(Y, Z).")
        assert tree.data == "start"

    def test_negation(self):
        tree = _parser().parse("unemployed(X) :- person(X), !employee(X).")
        assert tree.data == "start"

    def test_comparison(self):
        tree = _parser().parse("adult(X) :- person(X, Age), Age >= 18.")
        assert tree.data == "start"

    def test_funcall_in_body(self):
        tree = _parser().parse("total(X, S) :- val(X, V), S = fn:plus(V, 1).")
        assert tree.data == "start"

    def test_transform(self):
        tree = _parser().parse(
            "count(G, N) :- item(G, _) |> do fn:group_by(G), let N = fn:count()."
        )
        assert tree.data == "start"

    def test_builtin_predicate(self):
        tree = _parser().parse('ok(X) :- val(X), :string:starts_with(X, "hello").')
        assert tree.data == "start"


class TestGrammarDecls:
    def test_external_decl(self):
        tree = _parser().parse("Decl phrase_node(X, Y, Z) external.")
        assert tree.data == "start"

    def test_temporal_decl(self):
        tree = _parser().parse("Decl link(X, Y) temporal bound [/name, /name].")
        assert tree.data == "start"


class TestGrammarProgram:
    def test_multi_statement(self):
        tree = _parser().parse("""
            % Comment line
            edge("a", "b").
            edge("b", "c").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """)
        stmts = [c for c in tree.children if c is not None]
        assert len(stmts) == 4
```

**Step 3: Run tests**

```bash
cd ~/agentic-graph-rag && pip install lark
pytest pymangle/tests/test_grammar.py -v
```
Expected: 13 PASS (may need grammar tweaks — iterate until green)

**Step 4: Commit**

```bash
git add pymangle/pymangle/grammar.lark pymangle/tests/test_grammar.py
git commit -m "feat(pymangle): Lark grammar for Mangle/Datalog"
```

---

### Task 1.3: Parser (Lark → AST)

**Files:**
- Create: `pymangle/pymangle/parser.py`
- Create: `pymangle/tests/test_parser.py`

**Step 1: Write failing test**

```python
# pymangle/tests/test_parser.py
"""Tests for Mangle parser (Lark → AST)."""
from __future__ import annotations

import pytest

from pymangle.ast_nodes import Atom, Clause, Constant, NegAtom, Program, TermType, Variable
from pymangle.parser import parse, ParseError


class TestParseFacts:
    def test_string_args(self):
        prog = parse('edge("a", "b").')
        assert len(prog.facts) == 1
        fact = prog.facts[0]
        assert fact.head.predicate == "edge"
        assert fact.head.args == (
            Constant("a", TermType.STRING),
            Constant("b", TermType.STRING),
        )

    def test_name_constant(self):
        prog = parse("tool_for(/simple, \"vector_search\").")
        assert prog.facts[0].head.args[0] == Constant("simple", TermType.NAME)

    def test_number(self):
        prog = parse("score(42).")
        assert prog.facts[0].head.args[0] == Constant(42, TermType.NUMBER)

    def test_float(self):
        prog = parse("weight(3.14).")
        assert prog.facts[0].head.args[0] == Constant(3.14, TermType.FLOAT)


class TestParseRules:
    def test_simple_rule(self):
        prog = parse("reachable(X, Y) :- edge(X, Y).")
        assert len(prog.clauses) == 1
        clause = prog.clauses[0]
        assert clause.head.predicate == "reachable"
        assert len(clause.premises) == 1
        assert isinstance(clause.premises[0], Atom)

    def test_multi_premise(self):
        prog = parse("reachable(X, Z) :- edge(X, Y), edge(Y, Z).")
        assert len(prog.clauses[0].premises) == 2

    def test_negation(self):
        prog = parse("unemployed(X) :- person(X), !employee(X).")
        clause = prog.clauses[0]
        assert isinstance(clause.premises[1], NegAtom)
        assert clause.premises[1].atom.predicate == "employee"

    def test_variables_shared(self):
        prog = parse("path(X, Z) :- edge(X, Y), edge(Y, Z).")
        clause = prog.clauses[0]
        # Y appears in both premises
        assert clause.premises[0].args[1] == Variable("Y")
        assert clause.premises[1].args[0] == Variable("Y")


class TestParseTransform:
    def test_count_transform(self):
        prog = parse("total(G, N) :- item(G, _) |> do fn:group_by(G), let N = fn:count().")
        clause = prog.clauses[0]
        assert clause.transform is not None
        assert clause.transform.reducer.name == "fn:count"

    def test_sum_transform(self):
        prog = parse("total(G, S) :- item(G, V) |> do fn:group_by(G), let S = fn:sum(V).")
        clause = prog.clauses[0]
        assert clause.transform.reducer.name == "fn:sum"


class TestParseDecl:
    def test_external(self):
        prog = parse("Decl phrase_node(X, Y, Z) external.")
        assert len(prog.decls) == 1
        assert prog.decls[0].predicate == "phrase_node"
        assert prog.decls[0].arity == 3

    def test_temporal_with_bounds(self):
        prog = parse("Decl link(X, Y) temporal bound [/name, /name].")
        decl = prog.decls[0]
        assert "temporal" in [d.value for d in decl.descriptors]
        assert len(decl.bounds) == 1


class TestParseErrors:
    def test_missing_dot(self):
        with pytest.raises(ParseError):
            parse("edge(a, b)")

    def test_unclosed_paren(self):
        with pytest.raises(ParseError):
            parse("edge(a, b.")


class TestParseProgram:
    def test_full_program(self):
        prog = parse("""
            % Routing rules
            keyword(/relation, "связь").
            keyword(/relation, "between").

            match(Q, Cat) :- query_contains(Q, W), keyword(Cat, W).

            route_to(Q, "vector_search", 0.3) :- !match(Q, _).
        """)
        assert len(prog.facts) == 2
        assert len(prog.clauses) == 2
```

**Step 2: Run test — should FAIL**

```bash
pytest pymangle/tests/test_parser.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pymangle.parser'`

**Step 3: Implement parser**

```python
# pymangle/pymangle/parser.py
"""Parse Mangle source text into AST using Lark."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from lark import Lark, Transformer, v_args, exceptions as lark_exceptions

from pymangle.ast_nodes import (
    Atom,
    BoundDecl,
    Clause,
    Comparison,
    Constant,
    Decl,
    DeclDescriptor,
    FunCall,
    Interval,
    ListTerm,
    MapTerm,
    NegAtom,
    Program,
    StructTerm,
    TemporalAtom,
    Term,
    TermType,
    Transform,
    Variable,
)

logger = logging.getLogger(__name__)

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


class ParseError(Exception):
    """Raised when source text cannot be parsed."""


@v_args(inline=True)
class _ASTTransformer(Transformer):
    """Transform Lark parse tree into PyMangle AST."""

    def start(self, *statements):
        prog = Program()
        for stmt in statements:
            if stmt is None:
                continue
            if isinstance(stmt, Decl):
                prog.decls.append(stmt)
            elif isinstance(stmt, Clause):
                if stmt.premises:
                    prog.clauses.append(stmt)
                else:
                    prog.facts.append(stmt)
        return prog

    def statement(self, s):
        return s

    # --- Declarations ---

    def decl(self, pred, decl_args, *rest):
        descriptors = []
        bounds = []
        arity = len(decl_args) if isinstance(decl_args, list) else 1
        for item in rest:
            if isinstance(item, DeclDescriptor):
                descriptors.append(item)
            elif isinstance(item, BoundDecl):
                bounds.append(item)
        return Decl(
            predicate=str(pred),
            arity=arity,
            descriptors=tuple(descriptors),
            bounds=tuple(bounds),
        )

    def decl_args(self, *args):
        return list(args)

    def descriptor(self, token):
        return DeclDescriptor(str(token))

    def bound(self, type_list):
        return BoundDecl(types=tuple(type_list))

    def type_list(self, *types):
        return list(types)

    def type_expr(self, *tokens):
        return str(tokens[-1])

    # --- Facts ---

    def fact(self, atom, *rest):
        interval = rest[0] if rest else None
        return Clause(
            head=atom,
            premises=(),
            head_interval=interval,
        )

    # --- Rules ---

    def clause(self, head, *rest):
        interval = None
        body = None
        for item in rest:
            if isinstance(item, Interval):
                interval = item
            elif isinstance(item, tuple):
                body = item
        premises, transform = body if body else ((), None)
        return Clause(
            head=head,
            premises=tuple(premises),
            transform=transform,
            head_interval=interval,
        )

    def body(self, *parts):
        premises = []
        transform = None
        for p in parts:
            if isinstance(p, Transform):
                transform = p
            elif p is not None:
                premises.append(p)
        return (premises, transform)

    # --- Premises ---

    def premise(self, p):
        return p

    def neg_atom(self, atom):
        return NegAtom(atom=atom)

    def comparison(self, left, op, right):
        return Comparison(left=left, op=str(op), right=right)

    def temporal_atom(self, atom, interval):
        return TemporalAtom(atom=atom, interval=interval)

    def temporal_annot(self, start, end):
        return Interval(start=start, end=end)

    def time_expr(self, val):
        return val

    # --- Atoms ---

    def atom(self, pred, *args_list):
        args = args_list[0] if args_list else ()
        if isinstance(args, (list, tuple)):
            return Atom(predicate=str(pred), args=tuple(args))
        return Atom(predicate=str(pred), args=(args,) if args else ())

    def args(self, *terms):
        return list(terms)

    # --- Transform ---

    def transform(self, *parts):
        group_by = ()
        variable = None
        reducer = None
        for p in parts:
            if isinstance(p, tuple) and p[0] == "do":
                group_by = p[1]
            elif isinstance(p, tuple) and p[0] == "let":
                variable, reducer = p[1], p[2]
        return Transform(group_by=group_by, variable=variable, reducer=reducer)

    def do_clause(self, funcall):
        group_args = funcall.args if funcall.name == "fn:group_by" else ()
        return ("do", group_args)

    def let_clause(self, var, funcall):
        return ("let", Variable(str(var)), funcall)

    # --- Terms ---

    def variable(self, token):
        name = str(token)
        if name == "_":
            return Variable("_")
        return Variable(name)

    def constant(self, token):
        s = str(token)
        if s.startswith('"') and s.endswith('"'):
            return Constant(s[1:-1], TermType.STRING)
        if s.startswith("/"):
            return Constant(s[1:], TermType.NAME)
        if "." in s:
            return Constant(float(s), TermType.FLOAT)
        return Constant(int(s), TermType.NUMBER)

    def funcall(self, name, *args_list):
        args = args_list[0] if args_list else ()
        if isinstance(args, (list, tuple)):
            return FunCall(name=str(name), args=tuple(args))
        return FunCall(name=str(name), args=(args,) if args else ())

    def list_term(self, *args_list):
        args = args_list[0] if args_list else ()
        if isinstance(args, (list, tuple)):
            return ListTerm(elements=tuple(args))
        return ListTerm(elements=(args,) if args else ())

    def map_term(self, *args_list):
        args = args_list[0] if args_list else ()
        if isinstance(args, (list, tuple)):
            entries = tuple(
                (args[i], args[i + 1]) for i in range(0, len(args), 2)
            )
            return MapTerm(entries=entries)
        return MapTerm()

    def struct_term(self, *args_list):
        args = args_list[0] if args_list else ()
        if isinstance(args, (list, tuple)):
            fields = tuple(
                (str(args[i].value) if isinstance(args[i], Constant) else str(args[i]),
                 args[i + 1])
                for i in range(0, len(args), 2)
            )
            return StructTerm(fields=fields)
        return StructTerm()

    # --- Terminal handlers ---

    def PREDICATE(self, token):
        return str(token)

    def BUILTIN_PRED(self, token):
        return str(token)

    def VARIABLE(self, token):
        return str(token)

    def FN_NAME(self, token):
        return str(token)

    def NAME_CONST(self, token):
        return Constant(str(token)[1:], TermType.NAME)

    def STRING(self, token):
        return Constant(str(token)[1:-1], TermType.STRING)

    def NUMBER(self, token):
        return Constant(int(str(token)), TermType.NUMBER)

    def FLOAT(self, token):
        return Constant(float(str(token)), TermType.FLOAT)

    def ISO_DATE(self, token):
        return Constant(datetime.fromisoformat(str(token)), TermType.STRING)

    def ISO_DATETIME(self, token):
        return Constant(datetime.fromisoformat(str(token)), TermType.STRING)


def _get_parser() -> Lark:
    """Create Lark parser (cached)."""
    return Lark(
        _GRAMMAR_PATH.read_text(),
        start="start",
        parser="earley",
        transformer=None,
    )


_transformer = _ASTTransformer()


def parse(source: str) -> Program:
    """Parse Mangle source text into AST Program.

    Raises ParseError on invalid syntax.
    """
    try:
        tree = _get_parser().parse(source)
        return _transformer.transform(tree)
    except lark_exceptions.LarkError as e:
        raise ParseError(str(e)) from e


def load(path: str | Path) -> Program:
    """Load and parse a .mg file."""
    p = Path(path)
    return parse(p.read_text(encoding="utf-8"))
```

**Step 4: Run tests — iterate until green**

```bash
pytest pymangle/tests/test_parser.py -v
```
Expected: 15 PASS (grammar + transformer may need adjustments — iterate)

**Step 5: Export from `__init__.py`**

Add to `pymangle/pymangle/__init__.py`:

```python
from pymangle.parser import parse, load, ParseError
__all__ += ["parse", "load", "ParseError"]
```

**Step 6: Commit**

```bash
git add pymangle/pymangle/parser.py pymangle/tests/test_parser.py pymangle/pymangle/__init__.py
git commit -m "feat(pymangle): Lark parser — source text to AST"
```

---

## Phase 2: Unifier + FactStore

### Task 2.1: Unification

**Files:**
- Create: `pymangle/pymangle/unifier.py`
- Create: `pymangle/tests/test_unifier.py`

**Step 1: Write failing test**

```python
# pymangle/tests/test_unifier.py
"""Tests for term unification."""
from __future__ import annotations

from pymangle.ast_nodes import Atom, Constant, TermType, Variable
from pymangle.unifier import unify, apply_subst


class TestUnify:
    def test_var_to_const(self):
        subst = unify(Variable("X"), Constant("a", TermType.STRING), {})
        assert subst == {"X": Constant("a", TermType.STRING)}

    def test_const_to_const_same(self):
        a = Constant("a", TermType.STRING)
        subst = unify(a, a, {})
        assert subst == {}

    def test_const_to_const_diff(self):
        subst = unify(
            Constant("a", TermType.STRING),
            Constant("b", TermType.STRING),
            {},
        )
        assert subst is None

    def test_var_to_var(self):
        subst = unify(Variable("X"), Variable("Y"), {})
        assert subst is not None
        assert "X" in subst or "Y" in subst

    def test_bound_var(self):
        subst = unify(Variable("X"), Constant("b", TermType.STRING), {"X": Constant("a", TermType.STRING)})
        assert subst is None  # X already bound to "a", can't unify with "b"

    def test_wildcard(self):
        subst = unify(Variable("_"), Constant("anything", TermType.STRING), {})
        assert subst is not None  # _ unifies with anything, no binding added

    def test_atom_unify(self):
        pattern = Atom("edge", (Variable("X"), Variable("Y")))
        fact = Atom("edge", (Constant("a", TermType.STRING), Constant("b", TermType.STRING)))
        subst = unify(pattern, fact, {})
        assert subst == {
            "X": Constant("a", TermType.STRING),
            "Y": Constant("b", TermType.STRING),
        }


class TestApplySubst:
    def test_apply_to_var(self):
        result = apply_subst(Variable("X"), {"X": Constant("a", TermType.STRING)})
        assert result == Constant("a", TermType.STRING)

    def test_apply_to_atom(self):
        atom = Atom("edge", (Variable("X"), Variable("Y")))
        result = apply_subst(atom, {
            "X": Constant("a", TermType.STRING),
            "Y": Constant("b", TermType.STRING),
        })
        assert result == Atom("edge", (Constant("a", TermType.STRING), Constant("b", TermType.STRING)))
```

**Step 2: Implement unifier**

```python
# pymangle/pymangle/unifier.py
"""Substitution-based unification for Mangle terms."""
from __future__ import annotations

from pymangle.ast_nodes import Atom, Constant, FunCall, Term, Variable


def unify(t1: Term | Atom, t2: Term | Atom, subst: dict[str, Term]) -> dict[str, Term] | None:
    """Unify two terms under existing substitution.

    Returns extended substitution or None if unification fails.
    """
    # Resolve variables through existing substitution
    if isinstance(t1, Variable) and t1.name != "_" and t1.name in subst:
        return unify(subst[t1.name], t2, subst)
    if isinstance(t2, Variable) and t2.name != "_" and t2.name in subst:
        return unify(t1, subst[t2.name], subst)

    # Wildcard unifies with anything
    if isinstance(t1, Variable) and t1.name == "_":
        return dict(subst)
    if isinstance(t2, Variable) and t2.name == "_":
        return dict(subst)

    # Variable binds
    if isinstance(t1, Variable):
        new_subst = dict(subst)
        new_subst[t1.name] = t2
        return new_subst
    if isinstance(t2, Variable):
        new_subst = dict(subst)
        new_subst[t2.name] = t1
        return new_subst

    # Constants must be equal
    if isinstance(t1, Constant) and isinstance(t2, Constant):
        return dict(subst) if t1 == t2 else None

    # Atoms: same predicate + arity, unify args pairwise
    if isinstance(t1, Atom) and isinstance(t2, Atom):
        if t1.predicate != t2.predicate or t1.arity != t2.arity:
            return None
        current = dict(subst)
        for a1, a2 in zip(t1.args, t2.args):
            result = unify(a1, a2, current)
            if result is None:
                return None
            current = result
        return current

    # FunCall: same name, unify args
    if isinstance(t1, FunCall) and isinstance(t2, FunCall):
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        current = dict(subst)
        for a1, a2 in zip(t1.args, t2.args):
            result = unify(a1, a2, current)
            if result is None:
                return None
            current = result
        return current

    return None


def apply_subst(term: Term | Atom, subst: dict[str, Term]) -> Term | Atom:
    """Apply substitution to a term, resolving all variables."""
    if isinstance(term, Variable):
        if term.name == "_" or term.name not in subst:
            return term
        resolved = subst[term.name]
        # Chase chains: X → Y → value
        if isinstance(resolved, Variable):
            return apply_subst(resolved, subst)
        return resolved

    if isinstance(term, Constant):
        return term

    if isinstance(term, Atom):
        return Atom(
            predicate=term.predicate,
            args=tuple(apply_subst(a, subst) for a in term.args),
        )

    if isinstance(term, FunCall):
        return FunCall(
            name=term.name,
            args=tuple(apply_subst(a, subst) for a in term.args),
        )

    return term


def is_ground(term: Term | Atom) -> bool:
    """Check if term contains no variables."""
    if isinstance(term, Variable):
        return False
    if isinstance(term, Constant):
        return True
    if isinstance(term, Atom):
        return all(is_ground(a) for a in term.args)
    if isinstance(term, FunCall):
        return all(is_ground(a) for a in term.args)
    return True
```

**Step 3: Run tests**

```bash
pytest pymangle/tests/test_unifier.py -v
```
Expected: 9 PASS

**Step 4: Commit**

```bash
git add pymangle/pymangle/unifier.py pymangle/tests/test_unifier.py
git commit -m "feat(pymangle): substitution-based unification"
```

---

### Task 2.2: Indexed FactStore

**Files:**
- Create: `pymangle/pymangle/factstore.py`
- Create: `pymangle/tests/test_factstore.py`

**Step 1: Write failing test**

```python
# pymangle/tests/test_factstore.py
"""Tests for indexed fact storage."""
from __future__ import annotations

from pymangle.ast_nodes import Atom, Constant, TermType, Variable
from pymangle.factstore import IndexedFactStore


def _c(val: str) -> Constant:
    return Constant(val, TermType.STRING)


def _atom(pred: str, *args: str) -> Atom:
    return Atom(pred, tuple(_c(a) for a in args))


class TestFactStore:
    def test_add_new_fact(self):
        store = IndexedFactStore()
        assert store.add(_atom("edge", "a", "b")) is True
        assert len(store) == 1

    def test_add_duplicate(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        assert store.add(_atom("edge", "a", "b")) is False
        assert len(store) == 1

    def test_query_exact(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        store.add(_atom("edge", "a", "c"))
        store.add(_atom("node", "x"))

        results = list(store.query(Atom("edge", (Variable("X"), Variable("Y")))))
        assert len(results) == 2

    def test_query_first_arg_index(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        store.add(_atom("edge", "a", "c"))
        store.add(_atom("edge", "b", "d"))

        results = list(store.query(Atom("edge", (_c("a"), Variable("Y")))))
        assert len(results) == 2

    def test_query_no_match(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        results = list(store.query(Atom("node", (Variable("X"),))))
        assert len(results) == 0

    def test_merge(self):
        store1 = IndexedFactStore()
        store1.add(_atom("edge", "a", "b"))

        store2 = IndexedFactStore()
        store2.add(_atom("edge", "b", "c"))
        store2.add(_atom("edge", "a", "b"))  # duplicate

        new_facts = store1.merge(store2)
        assert len(store1) == 2
        assert len(new_facts) == 1  # only "b","c" is new

    def test_get_by_predicate(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        store.add(_atom("edge", "c", "d"))
        store.add(_atom("node", "x"))

        edges = list(store.get_by_predicate("edge"))
        assert len(edges) == 2

    def test_all_facts(self):
        store = IndexedFactStore()
        store.add(_atom("a", "1"))
        store.add(_atom("b", "2"))
        assert len(list(store.all_facts())) == 2
```

**Step 2: Implement factstore**

```python
# pymangle/pymangle/factstore.py
"""Indexed in-memory fact storage."""
from __future__ import annotations

import logging
from collections import defaultdict

from pymangle.ast_nodes import Atom, Constant, Variable
from pymangle.unifier import unify

logger = logging.getLogger(__name__)


class IndexedFactStore:
    """Hash-indexed fact store with predicate and first-arg indexes."""

    def __init__(self) -> None:
        self._facts: set[Atom] = set()
        self._by_predicate: dict[str, set[Atom]] = defaultdict(set)
        self._by_first_arg: dict[tuple[str, Constant], set[Atom]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._facts)

    def add(self, fact: Atom) -> bool:
        """Add a ground fact. Returns True if fact is new."""
        if fact in self._facts:
            return False
        self._facts.add(fact)
        self._by_predicate[fact.predicate].add(fact)
        if fact.args and isinstance(fact.args[0], Constant):
            self._by_first_arg[(fact.predicate, fact.args[0])].add(fact)
        return True

    def query(self, pattern: Atom) -> list[Atom]:
        """Find facts matching pattern (with variables as wildcards)."""
        # Use first-arg index if first arg is ground
        if pattern.args and isinstance(pattern.args[0], Constant):
            candidates = self._by_first_arg.get((pattern.predicate, pattern.args[0]), set())
        else:
            candidates = self._by_predicate.get(pattern.predicate, set())

        results = []
        for fact in candidates:
            if unify(pattern, fact, {}) is not None:
                results.append(fact)
        return results

    def merge(self, other: IndexedFactStore) -> IndexedFactStore:
        """Merge other store into self. Returns store with only NEW facts."""
        delta = IndexedFactStore()
        for fact in other._facts:
            if self.add(fact):
                delta.add(fact)
        return delta

    def get_by_predicate(self, predicate: str) -> list[Atom]:
        """Get all facts for a predicate."""
        return list(self._by_predicate.get(predicate, []))

    def all_facts(self) -> list[Atom]:
        """Get all facts."""
        return list(self._facts)

    def contains(self, fact: Atom) -> bool:
        """Check if exact fact exists."""
        return fact in self._facts
```

**Step 3: Run tests**

```bash
pytest pymangle/tests/test_factstore.py -v
```
Expected: 8 PASS

**Step 4: Commit**

```bash
git add pymangle/pymangle/factstore.py pymangle/tests/test_factstore.py
git commit -m "feat(pymangle): indexed in-memory fact store"
```

---

## Phase 3: Engine Core

### Task 3.1: Semi-Naive Bottom-Up Engine

**Files:**
- Create: `pymangle/pymangle/engine.py`
- Create: `pymangle/tests/test_engine.py`

**Step 1: Write failing tests**

```python
# pymangle/tests/test_engine.py
"""Tests for semi-naive bottom-up evaluation engine."""
from __future__ import annotations

import pytest

from pymangle.parser import parse
from pymangle.engine import eval_program, FactLimitError


class TestBasicEvaluation:
    def test_fact_only(self):
        """Facts are returned as-is."""
        store = eval_program(parse('edge("a", "b"). edge("b", "c").'))
        assert len(list(store.get_by_predicate("edge"))) == 2

    def test_single_rule(self):
        """Simple rule derives new facts."""
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c").
            path(X, Y) :- edge(X, Y).
        """))
        paths = store.get_by_predicate("path")
        assert len(paths) == 2

    def test_transitive_closure(self):
        """Recursive rule computes transitive closure."""
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c"). edge("c", "d").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """))
        results = store.get_by_predicate("reachable")
        # a→b, a→c, a→d, b→c, b→d, c→d = 6
        assert len(results) == 6

    def test_join(self):
        """Two-premise join."""
        store = eval_program(parse("""
            parent("alice", "bob").
            parent("bob", "carol").
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """))
        results = store.get_by_predicate("grandparent")
        assert len(results) == 1


class TestDeltaOptimization:
    def test_no_redundant_derivation(self):
        """Semi-naive avoids re-deriving known facts."""
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c"). edge("c", "a").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """))
        # Cycle: should still terminate with 9 reachable facts (3x3)
        results = store.get_by_predicate("reachable")
        assert len(results) == 9


class TestFactLimit:
    def test_limit_prevents_infinite(self):
        """Fact limit stops runaway evaluation."""
        with pytest.raises(FactLimitError):
            eval_program(
                parse("""
                    num(0).
                    num(X) :- num(Y), X = fn:plus(Y, 1).
                """),
                fact_limit=100,
            )


class TestMultipleRules:
    def test_multiple_rules_same_head(self):
        store = eval_program(parse("""
            cat("whiskers"). dog("rex").
            pet(X) :- cat(X).
            pet(X) :- dog(X).
        """))
        pets = store.get_by_predicate("pet")
        assert len(pets) == 2
```

**Step 2: Implement engine**

```python
# pymangle/pymangle/engine.py
"""Semi-naive bottom-up evaluation engine."""
from __future__ import annotations

import logging
from collections import defaultdict

from pymangle.ast_nodes import (
    Atom,
    Clause,
    Comparison,
    Constant,
    FunCall,
    NegAtom,
    Premise,
    Program,
    Variable,
)
from pymangle.factstore import IndexedFactStore
from pymangle.unifier import apply_subst, is_ground, unify

logger = logging.getLogger(__name__)


class FactLimitError(Exception):
    """Raised when derived fact limit is exceeded."""


def eval_program(
    program: Program,
    store: IndexedFactStore | None = None,
    fact_limit: int = 100_000,
) -> IndexedFactStore:
    """Evaluate a Mangle program using semi-naive bottom-up evaluation.

    Returns the fact store containing all derived facts.
    """
    if store is None:
        store = IndexedFactStore()

    # Load initial facts
    for fact_clause in program.facts:
        store.add(fact_clause.head)

    # Group rules by head predicate
    rules_by_pred: dict[str, list[Clause]] = defaultdict(list)
    for clause in program.clauses:
        rules_by_pred[clause.head.predicate].append(clause)

    if not rules_by_pred:
        return store

    total_derived = 0

    # Initial round: apply all rules
    delta = IndexedFactStore()
    for rules in rules_by_pred.values():
        for rule in rules:
            for fact in _eval_rule(rule, store, store):
                if store.add(fact):
                    delta.add(fact)
                    total_derived += 1
                    if total_derived > fact_limit:
                        raise FactLimitError(
                            f"Exceeded fact limit of {fact_limit}"
                        )

    # Delta iteration
    while len(delta) > 0:
        new_delta = IndexedFactStore()
        for rules in rules_by_pred.values():
            for rule in rules:
                # For each positive premise, create a delta variant
                for i, premise in enumerate(rule.premises):
                    if isinstance(premise, (NegAtom, Comparison)):
                        continue
                    for fact in _eval_rule_delta(rule, i, store, delta):
                        if store.add(fact):
                            new_delta.add(fact)
                            total_derived += 1
                            if total_derived > fact_limit:
                                raise FactLimitError(
                                    f"Exceeded fact limit of {fact_limit}"
                                )
        delta = new_delta

    logger.info("Evaluation complete: %d facts derived", total_derived)
    return store


def _eval_rule(
    rule: Clause,
    store: IndexedFactStore,
    search_store: IndexedFactStore,
) -> list[Atom]:
    """Evaluate a single rule against the store."""
    results = []
    for subst in _solve_premises(rule.premises, 0, {}, store, search_store):
        head = apply_subst(rule.head, subst)
        if isinstance(head, Atom) and is_ground(head):
            results.append(head)
    return results


def _eval_rule_delta(
    rule: Clause,
    delta_idx: int,
    store: IndexedFactStore,
    delta: IndexedFactStore,
) -> list[Atom]:
    """Evaluate rule with i-th premise using delta store."""
    results = []
    for subst in _solve_premises_delta(
        rule.premises, 0, delta_idx, {}, store, delta
    ):
        head = apply_subst(rule.head, subst)
        if isinstance(head, Atom) and is_ground(head):
            results.append(head)
    return results


def _solve_premises(
    premises: tuple[Premise, ...],
    idx: int,
    subst: dict[str, object],
    store: IndexedFactStore,
    search_store: IndexedFactStore,
) -> list[dict]:
    """Recursively solve premises left-to-right."""
    if idx >= len(premises):
        return [subst]

    premise = premises[idx]
    results = []

    if isinstance(premise, Atom):
        pattern = apply_subst(premise, subst)
        for fact in search_store.query(pattern):
            new_subst = unify(pattern, fact, subst)
            if new_subst is not None:
                results.extend(
                    _solve_premises(premises, idx + 1, new_subst, store, search_store)
                )

    elif isinstance(premise, NegAtom):
        pattern = apply_subst(premise.atom, subst)
        matches = store.query(pattern)
        if not matches:
            results.extend(
                _solve_premises(premises, idx + 1, subst, store, search_store)
            )

    elif isinstance(premise, Comparison):
        left = apply_subst(premise.left, subst)
        right = apply_subst(premise.right, subst)

        # Handle assignment: X = fn:plus(Y, 1)
        if isinstance(left, Variable) and is_ground(right):
            if isinstance(right, FunCall):
                from pymangle.builtins import eval_funcall
                val = eval_funcall(right)
                if val is not None:
                    new_subst = dict(subst)
                    new_subst[left.name] = val
                    results.extend(
                        _solve_premises(premises, idx + 1, new_subst, store, search_store)
                    )
            else:
                new_subst = dict(subst)
                new_subst[left.name] = right
                results.extend(
                    _solve_premises(premises, idx + 1, new_subst, store, search_store)
                )
        elif is_ground(left) and is_ground(right):
            if _eval_comparison(left, right, premise.op):
                results.extend(
                    _solve_premises(premises, idx + 1, subst, store, search_store)
                )

    return results


def _solve_premises_delta(
    premises: tuple[Premise, ...],
    idx: int,
    delta_idx: int,
    subst: dict[str, object],
    store: IndexedFactStore,
    delta: IndexedFactStore,
) -> list[dict]:
    """Solve premises with delta_idx-th premise using delta store."""
    if idx >= len(premises):
        return [subst]

    premise = premises[idx]
    results = []

    if isinstance(premise, Atom):
        pattern = apply_subst(premise, subst)
        # Use delta store for the delta premise, full store for others
        use_store = delta if idx == delta_idx else store
        for fact in use_store.query(pattern):
            new_subst = unify(pattern, fact, subst)
            if new_subst is not None:
                results.extend(
                    _solve_premises_delta(
                        premises, idx + 1, delta_idx, new_subst, store, delta
                    )
                )

    elif isinstance(premise, NegAtom):
        pattern = apply_subst(premise.atom, subst)
        matches = store.query(pattern)
        if not matches:
            results.extend(
                _solve_premises_delta(
                    premises, idx + 1, delta_idx, subst, store, delta
                )
            )

    elif isinstance(premise, Comparison):
        # Same as _solve_premises
        left = apply_subst(premise.left, subst)
        right = apply_subst(premise.right, subst)
        if isinstance(left, Variable) and is_ground(right):
            if isinstance(right, FunCall):
                from pymangle.builtins import eval_funcall
                val = eval_funcall(right)
                if val is not None:
                    new_subst = dict(subst)
                    new_subst[left.name] = val
                    results.extend(
                        _solve_premises_delta(
                            premises, idx + 1, delta_idx, new_subst, store, delta
                        )
                    )
            else:
                new_subst = dict(subst)
                new_subst[left.name] = right
                results.extend(
                    _solve_premises_delta(
                        premises, idx + 1, delta_idx, new_subst, store, delta
                    )
                )
        elif is_ground(left) and is_ground(right):
            if _eval_comparison(left, right, premise.op):
                results.extend(
                    _solve_premises_delta(
                        premises, idx + 1, delta_idx, subst, store, delta
                    )
                )

    return results


def _eval_comparison(left: object, right: object, op: str) -> bool:
    """Evaluate comparison between two ground terms."""
    lv = left.value if isinstance(left, Constant) else left
    rv = right.value if isinstance(right, Constant) else right

    if op == "==":
        return lv == rv
    if op == "!=":
        return lv != rv
    if op == "<":
        return lv < rv
    if op == "<=":
        return lv <= rv
    if op == ">":
        return lv > rv
    if op == ">=":
        return lv >= rv
    return False
```

**Step 3: Create minimal builtins stub (needed by engine)**

```python
# pymangle/pymangle/builtins.py
"""Built-in predicates and functions (stub — expanded in Phase 5)."""
from __future__ import annotations

from pymangle.ast_nodes import Constant, FunCall, Term, TermType


def eval_funcall(funcall: FunCall) -> Constant | None:
    """Evaluate a built-in function call. Returns None if unknown."""
    name = funcall.name
    args = funcall.args

    if name == "fn:plus" and len(args) == 2:
        a, b = args
        if isinstance(a, Constant) and isinstance(b, Constant):
            return Constant(a.value + b.value, TermType.NUMBER)

    return None
```

**Step 4: Run tests**

```bash
pytest pymangle/tests/test_engine.py -v
```
Expected: 6 PASS

**Step 5: Export from `__init__.py`**

Add to `pymangle/pymangle/__init__.py`:

```python
from pymangle.engine import eval_program, FactLimitError
__all__ += ["eval_program", "FactLimitError"]
```

**Step 6: Commit**

```bash
git add pymangle/pymangle/engine.py pymangle/pymangle/builtins.py pymangle/tests/test_engine.py pymangle/pymangle/__init__.py
git commit -m "feat(pymangle): semi-naive bottom-up evaluation engine"
```

---

## Phases 4-10: Summary

> **Note**: Phases 4-10 follow the same TDD pattern. Each is described at task-level below. Full code is provided for the critical path; remaining phases use the same conventions established in Phases 1-3.

---

## Phase 4: Stratification + Negation

### Task 4.1: Kosaraju SCC + Stratification

**Files:**
- Create: `pymangle/pymangle/analysis.py`
- Create: `pymangle/tests/test_analysis.py`

**Tests (8):**
- `test_no_negation_single_stratum` — all rules in one stratum
- `test_negation_two_strata` — negated pred in lower stratum
- `test_recursive_negation_rejected` — error on `p(X) :- !p(X)`
- `test_aggregation_creates_stratum` — transforms treated like negation
- `test_complex_dependency_graph` — 4+ predicates, 3 strata
- `test_independent_predicates` — unrelated predicates in same stratum
- `test_self_recursive_ok` — `reachable :- reachable` same stratum (positive)
- `test_stratified_eval_order` — facts computed in correct order

**Implementation:**
- `build_dependency_graph(clauses)` — edges with pos/neg labels
- `kosaraju_scc(graph)` — two-pass DFS
- `stratify(program) -> list[Stratum]` — validate no neg edge within SCC, topo sort
- Modify `eval_program()` to evaluate strata in order

**Commit:** `feat(pymangle): stratified evaluation with Kosaraju SCC`

---

## Phase 5: Builtins

### Task 5.1: Arithmetic, String, List, Map, Struct Operations

**Files:**
- Modify: `pymangle/pymangle/builtins.py`
- Create: `pymangle/tests/test_builtins.py`

**Tests (10):**
- Arithmetic: `fn:plus`, `fn:minus`, `fn:mult`, `fn:div`, `fn:float_div`
- String: `fn:string:concat`, `fn:string:len`, `:string:starts_with`, `:string:contains`
- List: `fn:list`, `fn:len`, `:list:member`
- Comparison builtins: `:time:le`, `:time:ge`

**Implementation:**
- `eval_funcall(funcall) -> Constant | None` — dispatch by `fn:` name
- `eval_builtin_pred(name, args) -> bool` — dispatch by `:` prefix

**Commit:** `feat(pymangle): 30+ built-in functions and predicates`

---

## Phase 6: Aggregation

### Task 6.1: Transform Pipeline

**Files:**
- Modify: `pymangle/pymangle/engine.py` (add post-fixpoint transforms)
- Create: `pymangle/tests/test_aggregation.py`

**Tests (5):**
- `test_count` — `fn:count()` per group
- `test_sum` — `fn:sum(V)` per group
- `test_collect` — `fn:collect(V)` into list
- `test_no_group_by` — global aggregation
- `test_multi_group_keys` — `fn:group_by(A, B)`

**Implementation:**
- After fixpoint, for rules with `transform`:
  - Collect all derived facts matching the rule body
  - Group by `transform.group_by` variables
  - Apply reducer (`fn:count`, `fn:sum`, `fn:avg`, `fn:min`, `fn:max`, `fn:collect`)
  - Emit aggregated head facts

**Commit:** `feat(pymangle): aggregation pipeline (group_by + reducers)`

---

## Phase 7: Temporal Reasoning

### Task 7.1: Temporal FactStore + Allen's Intervals

**Files:**
- Create: `pymangle/pymangle/temporal.py`
- Create: `pymangle/tests/test_temporal.py`

**Tests (12):**
- `test_add_temporal_fact` — store with interval
- `test_query_point_in_time` — facts valid at specific datetime
- `test_query_range` — facts overlapping with range
- `test_interval_coalescing` — overlapping intervals merged
- `test_allen_before` / `test_allen_meets` / `test_allen_overlaps` / `test_allen_during` / `test_allen_contains` / `test_allen_starts` / `test_allen_finishes` / `test_allen_equals`
- `test_temporal_rule_interval_intersection`

**Implementation:**
- `TemporalFactStore` — atom hash → interval list, coalescing, point/range queries
- `allen_before/meets/overlaps/during/contains/starts/finishes/equals` — 9 relations
- Integrate into engine: parallel temporal stores + temporal delta iteration

**Commit:** `feat(pymangle): temporal reasoning with Allen's interval algebra`

---

## Phase 8: Type System + External Predicates

### Task 8.1: Optional Type Bounds

**Files:**
- Create: `pymangle/pymangle/types.py`
- Extend: `pymangle/tests/test_engine.py`

**Tests (5):**
- `test_bounds_check_pass` — fact matches declared types
- `test_bounds_check_fail` — fact violates bounds → warning
- `test_no_decl_no_check` — no declaration → no checking
- `test_union_bounds` — multiple bound declarations (union)
- `test_mode_declaration` — input/output mode validation

**Implementation:**
- `TypeChecker.check_bounds(fact, decl) -> bool`
- Integrate into `eval_program()`: check each derived fact against decls

**Commit:** `feat(pymangle): optional type bounds checking`

### Task 8.2: External Predicate Protocol

**Files:**
- Create: `pymangle/pymangle/external.py`
- Create: `pymangle/tests/test_external.py`

**Tests (5):**
- `test_external_basic_query` — mock external returns facts
- `test_external_with_input` — bound args passed as inputs
- `test_external_no_results` — empty result handled
- `test_external_filter_pushdown` — filters passed to callback
- `test_external_in_rule` — external predicate used in rule body

**Implementation:**
```python
class ExternalPredicate(Protocol):
    def query(self, inputs: list[Constant], filters: list) -> Iterator[list[Constant]]: ...

def register_external(program, predicate: str, callback: ExternalPredicate) -> None: ...
```
- Engine checks external predicates when no facts in store
- Pass bound arguments as `inputs`, unbound as `filters`

**Commit:** `feat(pymangle): external predicate protocol with filter pushdown`

---

## Phase 9: Integration Layer

### Task 9.1: Neo4j Bridge

**Files:**
- Create: `agentic_graph_rag/reasoning/__init__.py`
- Create: `agentic_graph_rag/reasoning/neo4j_bridge.py`
- Create: `agentic_graph_rag/reasoning/reasoning_engine.py`
- Create: `tests/test_reasoning_bridge.py`

**Tests (2):**
- `test_neo4j_external_predicate` — mock driver, verify Cypher executed
- `test_filter_pushdown` — bound args passed to Cypher params

**Implementation:**
- `Neo4jExternalPredicate(driver, cypher_template)` — implements `ExternalPredicate`
- `ReasoningEngine(rules_dir)` — loads .mg files, registers externals, provides `classify_query()`, `infer_connections()`, `check_access()`

**Commit:** `feat(agentic-graph-rag): Neo4j bridge + ReasoningEngine facade`

### Task 9.2: Rule Files

**Files:**
- Create: `agentic_graph_rag/reasoning/rules/routing.mg`
- Create: `agentic_graph_rag/reasoning/rules/graph.mg`
- Create: `agentic_graph_rag/reasoning/rules/access.mg`
- Create: `tests/test_reasoning_routing.py`
- Create: `tests/test_reasoning_access.py`

**Tests (13):**
- Routing (8): relation, multi_hop, global, temporal, simple default, multiple keywords, confidence, bilingual
- Access (5): permit, deny overrides, role inheritance, visible_passage filtering, denied_query audit

**Implementation:**
- Copy rules from design doc (Sections 3.1, 3.2, 3.3)
- Inject `query_contains` as external predicate (simple string tokenizer)

**Commit:** `feat(agentic-graph-rag): declarative routing, graph, and access rules`

---

## Phase 10: Wiring + Benchmark

### Task 10.1: Router Integration

**Files:**
- Modify: `agentic_graph_rag/agent/router.py` (add Mangle path)
- Modify: `agentic_graph_rag/agent/retrieval_agent.py` (pass reasoning engine)

**Tests (3):**
- `test_mangle_router_used_when_available` — Mangle result returned
- `test_fallback_to_patterns` — Mangle confidence < 0.5 → pattern fallback
- `test_no_mangle_backward_compat` — `reasoning=None` → existing behavior

**Implementation:**
- Add `reasoning: ReasoningEngine | None = None` param to `classify_query()`
- Mangle first → fallback to patterns if confidence < 0.5

**Commit:** `feat(agentic-graph-rag): integrate Mangle reasoning into query router`

### Task 10.2: Benchmark Mode

**Files:**
- Modify: `benchmark/runner.py` (add `agent_mangle` mode)
- Modify: `benchmark/questions.json` (no changes needed — same questions)

**Tests (2):**
- `test_benchmark_mangle_mode_runs` — smoke test
- `test_benchmark_mangle_vs_pattern` — comparative output

**Implementation:**
- Add `"agent_mangle"` to benchmark modes
- Use `ReasoningEngine` for routing in this mode

**Commit:** `feat(agentic-graph-rag): benchmark mode for Mangle-based routing`

### Task 10.3: Streamlit Tab (Optional)

**Files:**
- Modify: `ui/streamlit_app.py` (add "Reasoning" tab)

**Implementation:**
- Tab 7: "Reasoning"
- Text area to edit .mg rules
- Input field for test query
- Display: routing decision, graph inferences, access check
- Stratification visualization (mermaid or text)

**Commit:** `feat(agentic-graph-rag): Streamlit Reasoning tab for rule testing`

---

## Summary

| Phase | Tasks | Tests | Commits |
|-------|-------|-------|---------|
| 1. AST + Parser | 3 | 15 | 3 |
| 2. Unifier + FactStore | 2 | 9+8=17 | 2 |
| 3. Engine Core | 1 | 6 | 1 |
| 4. Stratification | 1 | 8 | 1 |
| 5. Builtins | 1 | 10 | 1 |
| 6. Aggregation | 1 | 5 | 1 |
| 7. Temporal | 1 | 12 | 1 |
| 8. Types + External | 2 | 5+5=10 | 2 |
| 9. Integration | 2 | 2+13=15 | 2 |
| 10. Wiring + Bench | 3 | 3+2=5 | 3 |
| **Total** | **17** | **~103** | **17** |

**Parallelizable after Phase 3:** Phases 4, 5, 7 can run concurrently (independent).
