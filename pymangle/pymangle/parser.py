"""Parse Mangle source text into AST using Lark."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from lark import Lark, Transformer, exceptions as lark_exceptions, v_args

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

    def agg_transform(self, *parts):
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
