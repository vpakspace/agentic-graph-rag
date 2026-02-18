"""Built-in functions and predicates for Mangle programs."""
from __future__ import annotations

import math
import re

from pymangle.ast_nodes import Constant, FunCall, ListTerm, MapTerm, StructTerm, Term, TermType


def _numeric_result(value: int | float) -> Constant:
    """Wrap a numeric result in the correct Constant type."""
    if isinstance(value, float):
        return Constant(value, TermType.FLOAT)
    return Constant(value, TermType.NUMBER)


def _two_nums(args: tuple[Term, ...]) -> tuple[int | float, int | float] | None:
    """Extract two numeric values from args, or None."""
    if len(args) != 2:
        return None
    a, b = args
    if isinstance(a, Constant) and isinstance(b, Constant):
        av, bv = a.value, b.value
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            return av, bv
    return None


def _one_num(args: tuple[Term, ...]) -> int | float | None:
    """Extract single numeric value from args, or None."""
    if len(args) != 1:
        return None
    a = args[0]
    if isinstance(a, Constant) and isinstance(a.value, (int, float)):
        return a.value
    return None


def _one_str(args: tuple[Term, ...]) -> str | None:
    """Extract single string value from args, or None."""
    if len(args) != 1:
        return None
    a = args[0]
    if isinstance(a, Constant):
        return str(a.value)
    return None


def _two_strs(args: tuple[Term, ...]) -> tuple[str, str] | None:
    """Extract two string values from args, or None."""
    if len(args) != 2:
        return None
    a, b = args
    if isinstance(a, Constant) and isinstance(b, Constant):
        return str(a.value), str(b.value)
    return None


def eval_funcall(funcall: FunCall) -> Constant | ListTerm | MapTerm | StructTerm | None:
    """Evaluate a built-in function call. Returns None if unknown."""
    name = funcall.name
    args = funcall.args

    # --- Arithmetic ---
    if name == "fn:plus":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] + pair[1])

    elif name == "fn:minus":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] - pair[1])

    elif name == "fn:mult":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] * pair[1])

    elif name == "fn:div":
        pair = _two_nums(args)
        if pair and pair[1] != 0:
            return _numeric_result(int(pair[0] // pair[1]))

    elif name == "fn:float_div":
        pair = _two_nums(args)
        if pair and pair[1] != 0:
            return Constant(pair[0] / pair[1], TermType.FLOAT)

    elif name == "fn:mod":
        pair = _two_nums(args)
        if pair and pair[1] != 0:
            return _numeric_result(pair[0] % pair[1])

    elif name == "fn:abs":
        val = _one_num(args)
        if val is not None:
            return _numeric_result(abs(val))

    elif name == "fn:pow":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] ** pair[1])

    elif name == "fn:sqrt":
        val = _one_num(args)
        if val is not None and val >= 0:
            return Constant(math.sqrt(val), TermType.FLOAT)

    elif name == "fn:floor":
        val = _one_num(args)
        if val is not None:
            return Constant(int(math.floor(val)), TermType.NUMBER)

    elif name == "fn:ceil":
        val = _one_num(args)
        if val is not None:
            return Constant(int(math.ceil(val)), TermType.NUMBER)

    elif name == "fn:round":
        val = _one_num(args)
        if val is not None:
            return Constant(int(round(val)), TermType.NUMBER)

    # --- String functions ---
    elif name == "fn:string:concat":
        pair = _two_strs(args)
        if pair:
            return Constant(pair[0] + pair[1], TermType.STRING)

    elif name == "fn:string:len":
        val = _one_str(args)
        if val is not None:
            return Constant(len(val), TermType.NUMBER)

    elif name == "fn:string:uppercase":
        val = _one_str(args)
        if val is not None:
            return Constant(val.upper(), TermType.STRING)

    elif name == "fn:string:lowercase":
        val = _one_str(args)
        if val is not None:
            return Constant(val.lower(), TermType.STRING)

    elif name == "fn:string:substring":
        if len(args) == 3 and all(isinstance(a, Constant) for a in args):
            s = str(args[0].value)
            start = args[1].value
            end = args[2].value
            if isinstance(start, int) and isinstance(end, int):
                return Constant(s[start:end], TermType.STRING)

    elif name == "fn:string:replace":
        if len(args) == 3 and all(isinstance(a, Constant) for a in args):
            s = str(args[0].value)
            old = str(args[1].value)
            new = str(args[2].value)
            return Constant(s.replace(old, new), TermType.STRING)

    elif name == "fn:string:split":
        pair = _two_strs(args)
        if pair:
            parts = pair[0].split(pair[1])
            elements = tuple(Constant(p, TermType.STRING) for p in parts)
            return ListTerm(elements)

    elif name == "fn:string:trim":
        val = _one_str(args)
        if val is not None:
            return Constant(val.strip(), TermType.STRING)

    # --- List functions ---
    elif name == "fn:list":
        return ListTerm(args)

    elif name == "fn:len":
        if len(args) == 1 and isinstance(args[0], ListTerm):
            return Constant(len(args[0].elements), TermType.NUMBER)

    elif name == "fn:list:nth":
        if len(args) == 2 and isinstance(args[0], ListTerm) and isinstance(args[1], Constant):
            idx = args[1].value
            if isinstance(idx, int) and 0 <= idx < len(args[0].elements):
                elem = args[0].elements[idx]
                if isinstance(elem, (Constant, ListTerm, MapTerm, StructTerm)):
                    return elem

    elif name == "fn:list:append":
        if len(args) == 2 and isinstance(args[0], ListTerm):
            return ListTerm(args[0].elements + (args[1],))

    elif name == "fn:list:reverse":
        if len(args) == 1 and isinstance(args[0], ListTerm):
            return ListTerm(tuple(reversed(args[0].elements)))

    elif name == "fn:list:sort":
        if len(args) == 1 and isinstance(args[0], ListTerm):
            elems = args[0].elements
            if all(isinstance(e, Constant) for e in elems):
                sorted_elems = sorted(elems, key=lambda e: e.value if isinstance(e, Constant) else 0)
                return ListTerm(tuple(sorted_elems))

    elif name == "fn:list:head":
        if len(args) == 1 and isinstance(args[0], ListTerm) and args[0].elements:
            elem = args[0].elements[0]
            if isinstance(elem, (Constant, ListTerm, MapTerm, StructTerm)):
                return elem

    elif name == "fn:list:tail":
        if len(args) == 1 and isinstance(args[0], ListTerm) and args[0].elements:
            return ListTerm(args[0].elements[1:])

    # --- Map functions ---
    elif name == "fn:map":
        # fn:map(k1, v1, k2, v2, ...) → MapTerm
        if len(args) % 2 == 0:
            entries = tuple((args[i], args[i + 1]) for i in range(0, len(args), 2))
            return MapTerm(entries)

    elif name == "fn:map:get":
        if len(args) == 2 and isinstance(args[0], MapTerm):
            key = args[1]
            for k, v in args[0].entries:
                if k == key and isinstance(v, (Constant, ListTerm, MapTerm, StructTerm)):
                    return v

    elif name == "fn:map:put":
        if len(args) == 3 and isinstance(args[0], MapTerm):
            key, val = args[1], args[2]
            new_entries = tuple((k, v) for k, v in args[0].entries if k != key) + ((key, val),)
            return MapTerm(new_entries)

    elif name == "fn:map:size":
        if len(args) == 1 and isinstance(args[0], MapTerm):
            return Constant(len(args[0].entries), TermType.NUMBER)

    # --- Struct functions ---
    elif name == "fn:struct:get":
        if len(args) == 2 and isinstance(args[0], StructTerm) and isinstance(args[1], Constant):
            field_name = str(args[1].value)
            for name_f, val in args[0].fields:
                if name_f == field_name and isinstance(val, (Constant, ListTerm, MapTerm, StructTerm)):
                    return val

    elif name == "fn:struct:set":
        if len(args) == 3 and isinstance(args[0], StructTerm) and isinstance(args[1], Constant):
            field_name = str(args[1].value)
            new_fields = tuple((n, v) if n != field_name else (n, args[2]) for n, v in args[0].fields)
            return StructTerm(new_fields)

    # --- Type checking functions ---
    elif name == "fn:type":
        if len(args) == 1:
            a = args[0]
            if isinstance(a, Constant):
                return Constant(a.type.value, TermType.STRING)
            if isinstance(a, ListTerm):
                return Constant("list", TermType.STRING)
            if isinstance(a, MapTerm):
                return Constant("map", TermType.STRING)
            if isinstance(a, StructTerm):
                return Constant("struct", TermType.STRING)

    return None


def eval_builtin_pred(name: str, args: list[Term]) -> bool:
    """Evaluate a built-in predicate (: prefix). Returns False if unknown."""

    # --- String predicates ---
    if name == ":string:starts_with":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return str(args[0].value).startswith(str(args[1].value))

    elif name == ":string:contains":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return str(args[1].value) in str(args[0].value)

    elif name == ":string:ends_with":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return str(args[0].value).endswith(str(args[1].value))

    elif name == ":string:matches":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return bool(re.search(str(args[1].value), str(args[0].value)))

    # --- List predicates ---
    elif name == ":list:member":
        if len(args) == 2 and isinstance(args[1], ListTerm):
            return args[0] in args[1].elements

    elif name == ":list:empty":
        if len(args) == 1 and isinstance(args[0], ListTerm):
            return len(args[0].elements) == 0

    # --- Type predicates ---
    elif name == ":is_number":
        if len(args) == 1 and isinstance(args[0], Constant):
            return args[0].type in (TermType.NUMBER, TermType.FLOAT)

    elif name == ":is_string":
        if len(args) == 1 and isinstance(args[0], Constant):
            return args[0].type == TermType.STRING

    elif name == ":is_list":
        return len(args) == 1 and isinstance(args[0], ListTerm)

    elif name == ":is_map":
        return len(args) == 1 and isinstance(args[0], MapTerm)

    elif name == ":is_struct":
        return len(args) == 1 and isinstance(args[0], StructTerm)

    return False
