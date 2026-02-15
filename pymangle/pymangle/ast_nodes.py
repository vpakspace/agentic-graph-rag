"""AST nodes for Mangle/Datalog programs."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


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
