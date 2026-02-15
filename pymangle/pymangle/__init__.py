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

from pymangle.engine import FactLimitError, eval_program
from pymangle.parser import ParseError, load, parse

__all__ = [
    "FactLimitError",
    "Atom",
    "Clause",
    "Comparison",
    "Constant",
    "Decl",
    "FunCall",
    "Interval",
    "NegAtom",
    "ParseError",
    "Program",
    "TemporalAtom",
    "Transform",
    "Variable",
    "eval_program",
    "load",
    "parse",
]
