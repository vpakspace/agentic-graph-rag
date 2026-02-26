"""Tests for built-in functions and predicates."""
from __future__ import annotations

from pymangle.ast_nodes import Constant, FunCall, ListTerm, MapTerm, StructTerm, TermType
from pymangle.builtins import eval_builtin_pred, eval_funcall


# ── Helpers edge cases ──────────────────────────────────────────────


class TestHelperEdgeCases:
    """Cover _two_nums, _one_num, _one_str, _two_strs guard paths."""

    def test_two_nums_wrong_count(self):
        result = eval_funcall(FunCall("fn:plus", (Constant(1, TermType.NUMBER),)))
        assert result is None

    def test_two_nums_non_numeric(self):
        result = eval_funcall(FunCall("fn:plus", (Constant("a", TermType.STRING), Constant(1, TermType.NUMBER))))
        assert result is None

    def test_one_num_wrong_count(self):
        result = eval_funcall(FunCall("fn:abs", ()))
        assert result is None

    def test_one_num_non_numeric(self):
        result = eval_funcall(FunCall("fn:abs", (Constant("x", TermType.STRING),)))
        assert result is None

    def test_one_str_wrong_count(self):
        result = eval_funcall(FunCall("fn:string:len", ()))
        assert result is None

    def test_two_strs_wrong_count(self):
        result = eval_funcall(FunCall("fn:string:concat", (Constant("a", TermType.STRING),)))
        assert result is None


# ── Arithmetic ──────────────────────────────────────────────────────


class TestArithmetic:
    def test_plus(self):
        result = eval_funcall(FunCall("fn:plus", (Constant(3, TermType.NUMBER), Constant(4, TermType.NUMBER))))
        assert result == Constant(7, TermType.NUMBER)

    def test_minus(self):
        result = eval_funcall(FunCall("fn:minus", (Constant(10, TermType.NUMBER), Constant(3, TermType.NUMBER))))
        assert result == Constant(7, TermType.NUMBER)

    def test_mult(self):
        result = eval_funcall(FunCall("fn:mult", (Constant(3, TermType.NUMBER), Constant(4, TermType.NUMBER))))
        assert result == Constant(12, TermType.NUMBER)

    def test_div(self):
        result = eval_funcall(FunCall("fn:div", (Constant(10, TermType.NUMBER), Constant(3, TermType.NUMBER))))
        assert result == Constant(3, TermType.NUMBER)

    def test_float_div(self):
        result = eval_funcall(FunCall("fn:float_div", (Constant(10, TermType.NUMBER), Constant(4, TermType.NUMBER))))
        assert result == Constant(2.5, TermType.FLOAT)

    def test_float_result(self):
        result = eval_funcall(FunCall("fn:plus", (Constant(1.5, TermType.FLOAT), Constant(2, TermType.NUMBER))))
        assert result == Constant(3.5, TermType.FLOAT)

    def test_div_by_zero(self):
        result = eval_funcall(FunCall("fn:div", (Constant(10, TermType.NUMBER), Constant(0, TermType.NUMBER))))
        assert result is None

    def test_float_div_by_zero(self):
        result = eval_funcall(FunCall("fn:float_div", (Constant(10, TermType.NUMBER), Constant(0, TermType.NUMBER))))
        assert result is None

    def test_mod(self):
        result = eval_funcall(FunCall("fn:mod", (Constant(10, TermType.NUMBER), Constant(3, TermType.NUMBER))))
        assert result == Constant(1, TermType.NUMBER)

    def test_mod_by_zero(self):
        result = eval_funcall(FunCall("fn:mod", (Constant(10, TermType.NUMBER), Constant(0, TermType.NUMBER))))
        assert result is None

    def test_abs(self):
        result = eval_funcall(FunCall("fn:abs", (Constant(-5, TermType.NUMBER),)))
        assert result == Constant(5, TermType.NUMBER)

    def test_abs_float(self):
        result = eval_funcall(FunCall("fn:abs", (Constant(-3.14, TermType.FLOAT),)))
        assert result == Constant(3.14, TermType.FLOAT)

    def test_pow(self):
        result = eval_funcall(FunCall("fn:pow", (Constant(2, TermType.NUMBER), Constant(10, TermType.NUMBER))))
        assert result == Constant(1024, TermType.NUMBER)

    def test_sqrt(self):
        result = eval_funcall(FunCall("fn:sqrt", (Constant(16, TermType.NUMBER),)))
        assert result == Constant(4.0, TermType.FLOAT)

    def test_sqrt_negative(self):
        result = eval_funcall(FunCall("fn:sqrt", (Constant(-1, TermType.NUMBER),)))
        assert result is None

    def test_floor(self):
        result = eval_funcall(FunCall("fn:floor", (Constant(3.7, TermType.FLOAT),)))
        assert result == Constant(3, TermType.NUMBER)

    def test_ceil(self):
        result = eval_funcall(FunCall("fn:ceil", (Constant(3.2, TermType.FLOAT),)))
        assert result == Constant(4, TermType.NUMBER)

    def test_round(self):
        result = eval_funcall(FunCall("fn:round", (Constant(3.5, TermType.FLOAT),)))
        assert result == Constant(4, TermType.NUMBER)

    def test_round_down(self):
        result = eval_funcall(FunCall("fn:round", (Constant(3.4, TermType.FLOAT),)))
        assert result == Constant(3, TermType.NUMBER)


# ── String functions ────────────────────────────────────────────────


class TestString:
    def test_concat(self):
        result = eval_funcall(FunCall("fn:string:concat", (Constant("hello", TermType.STRING), Constant(" world", TermType.STRING))))
        assert result == Constant("hello world", TermType.STRING)

    def test_len(self):
        result = eval_funcall(FunCall("fn:string:len", (Constant("hello", TermType.STRING),)))
        assert result == Constant(5, TermType.NUMBER)

    def test_uppercase(self):
        result = eval_funcall(FunCall("fn:string:uppercase", (Constant("hello", TermType.STRING),)))
        assert result == Constant("HELLO", TermType.STRING)

    def test_lowercase(self):
        result = eval_funcall(FunCall("fn:string:lowercase", (Constant("HELLO", TermType.STRING),)))
        assert result == Constant("hello", TermType.STRING)

    def test_substring(self):
        result = eval_funcall(FunCall("fn:string:substring", (
            Constant("hello world", TermType.STRING),
            Constant(0, TermType.NUMBER),
            Constant(5, TermType.NUMBER),
        )))
        assert result == Constant("hello", TermType.STRING)

    def test_substring_wrong_arg_count(self):
        result = eval_funcall(FunCall("fn:string:substring", (Constant("hello", TermType.STRING),)))
        assert result is None

    def test_substring_non_int_indices(self):
        result = eval_funcall(FunCall("fn:string:substring", (
            Constant("hello", TermType.STRING),
            Constant("a", TermType.STRING),
            Constant("b", TermType.STRING),
        )))
        assert result is None

    def test_replace(self):
        result = eval_funcall(FunCall("fn:string:replace", (
            Constant("hello world", TermType.STRING),
            Constant("world", TermType.STRING),
            Constant("earth", TermType.STRING),
        )))
        assert result == Constant("hello earth", TermType.STRING)

    def test_replace_wrong_arg_count(self):
        result = eval_funcall(FunCall("fn:string:replace", (Constant("hello", TermType.STRING),)))
        assert result is None

    def test_split(self):
        result = eval_funcall(FunCall("fn:string:split", (
            Constant("a,b,c", TermType.STRING),
            Constant(",", TermType.STRING),
        )))
        assert isinstance(result, ListTerm)
        assert len(result.elements) == 3
        assert result.elements[0] == Constant("a", TermType.STRING)

    def test_trim(self):
        result = eval_funcall(FunCall("fn:string:trim", (Constant("  hello  ", TermType.STRING),)))
        assert result == Constant("hello", TermType.STRING)

    def test_starts_with(self):
        assert eval_builtin_pred(":string:starts_with", [Constant("hello", TermType.STRING), Constant("hel", TermType.STRING)]) is True
        assert eval_builtin_pred(":string:starts_with", [Constant("hello", TermType.STRING), Constant("xyz", TermType.STRING)]) is False

    def test_contains(self):
        assert eval_builtin_pred(":string:contains", [Constant("hello world", TermType.STRING), Constant("lo wo", TermType.STRING)]) is True
        assert eval_builtin_pred(":string:contains", [Constant("hello", TermType.STRING), Constant("xyz", TermType.STRING)]) is False

    def test_ends_with(self):
        assert eval_builtin_pred(":string:ends_with", [Constant("hello", TermType.STRING), Constant("llo", TermType.STRING)]) is True
        assert eval_builtin_pred(":string:ends_with", [Constant("hello", TermType.STRING), Constant("xyz", TermType.STRING)]) is False

    def test_matches(self):
        assert eval_builtin_pred(":string:matches", [Constant("hello123", TermType.STRING), Constant(r"\d+", TermType.STRING)]) is True
        assert eval_builtin_pred(":string:matches", [Constant("hello", TermType.STRING), Constant(r"\d+", TermType.STRING)]) is False


# ── List functions ──────────────────────────────────────────────────


class TestList:
    def test_list_constructor(self):
        result = eval_funcall(FunCall("fn:list", (Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER))))
        assert isinstance(result, ListTerm)
        assert len(result.elements) == 2

    def test_len(self):
        lst = ListTerm((Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER), Constant(3, TermType.NUMBER)))
        result = eval_funcall(FunCall("fn:len", (lst,)))
        assert result == Constant(3, TermType.NUMBER)

    def test_nth(self):
        lst = ListTerm((Constant("a", TermType.STRING), Constant("b", TermType.STRING)))
        result = eval_funcall(FunCall("fn:list:nth", (lst, Constant(1, TermType.NUMBER))))
        assert result == Constant("b", TermType.STRING)

    def test_nth_out_of_range(self):
        lst = ListTerm((Constant("a", TermType.STRING),))
        result = eval_funcall(FunCall("fn:list:nth", (lst, Constant(5, TermType.NUMBER))))
        assert result is None

    def test_nth_negative_index(self):
        lst = ListTerm((Constant("a", TermType.STRING),))
        result = eval_funcall(FunCall("fn:list:nth", (lst, Constant(-1, TermType.NUMBER))))
        assert result is None

    def test_append(self):
        lst = ListTerm((Constant(1, TermType.NUMBER),))
        result = eval_funcall(FunCall("fn:list:append", (lst, Constant(2, TermType.NUMBER))))
        assert isinstance(result, ListTerm)
        assert len(result.elements) == 2
        assert result.elements[1] == Constant(2, TermType.NUMBER)

    def test_reverse(self):
        lst = ListTerm((Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER), Constant(3, TermType.NUMBER)))
        result = eval_funcall(FunCall("fn:list:reverse", (lst,)))
        assert isinstance(result, ListTerm)
        assert result.elements[0] == Constant(3, TermType.NUMBER)
        assert result.elements[2] == Constant(1, TermType.NUMBER)

    def test_sort(self):
        lst = ListTerm((Constant(3, TermType.NUMBER), Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER)))
        result = eval_funcall(FunCall("fn:list:sort", (lst,)))
        assert isinstance(result, ListTerm)
        assert result.elements == (Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER), Constant(3, TermType.NUMBER))

    def test_head(self):
        lst = ListTerm((Constant("first", TermType.STRING), Constant("second", TermType.STRING)))
        result = eval_funcall(FunCall("fn:list:head", (lst,)))
        assert result == Constant("first", TermType.STRING)

    def test_head_empty(self):
        result = eval_funcall(FunCall("fn:list:head", (ListTerm(()),)))
        assert result is None

    def test_tail(self):
        lst = ListTerm((Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER), Constant(3, TermType.NUMBER)))
        result = eval_funcall(FunCall("fn:list:tail", (lst,)))
        assert isinstance(result, ListTerm)
        assert len(result.elements) == 2
        assert result.elements[0] == Constant(2, TermType.NUMBER)

    def test_tail_empty(self):
        result = eval_funcall(FunCall("fn:list:tail", (ListTerm(()),)))
        assert result is None

    def test_member(self):
        lst = ListTerm((Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER)))
        assert eval_builtin_pred(":list:member", [Constant(1, TermType.NUMBER), lst]) is True
        assert eval_builtin_pred(":list:member", [Constant(9, TermType.NUMBER), lst]) is False

    def test_empty(self):
        assert eval_builtin_pred(":list:empty", [ListTerm(())]) is True
        assert eval_builtin_pred(":list:empty", [ListTerm((Constant(1, TermType.NUMBER),))]) is False


# ── Map functions ───────────────────────────────────────────────────


class TestMap:
    def test_map_constructor(self):
        result = eval_funcall(FunCall("fn:map", (
            Constant("k1", TermType.STRING), Constant("v1", TermType.STRING),
            Constant("k2", TermType.STRING), Constant("v2", TermType.STRING),
        )))
        assert isinstance(result, MapTerm)
        assert len(result.entries) == 2

    def test_map_odd_args(self):
        """Odd number of args → None (not even pairs)."""
        result = eval_funcall(FunCall("fn:map", (Constant("k", TermType.STRING),)))
        assert result is None

    def test_map_get(self):
        m = MapTerm((
            (Constant("name", TermType.STRING), Constant("alice", TermType.STRING)),
            (Constant("age", TermType.STRING), Constant(30, TermType.NUMBER)),
        ))
        result = eval_funcall(FunCall("fn:map:get", (m, Constant("name", TermType.STRING))))
        assert result == Constant("alice", TermType.STRING)

    def test_map_get_missing_key(self):
        m = MapTerm(((Constant("k", TermType.STRING), Constant("v", TermType.STRING)),))
        result = eval_funcall(FunCall("fn:map:get", (m, Constant("missing", TermType.STRING))))
        assert result is None

    def test_map_put(self):
        m = MapTerm(((Constant("k", TermType.STRING), Constant("old", TermType.STRING)),))
        result = eval_funcall(FunCall("fn:map:put", (m, Constant("k", TermType.STRING), Constant("new", TermType.STRING))))
        assert isinstance(result, MapTerm)
        # Key replaced, not duplicated
        assert len(result.entries) == 1
        assert result.entries[0][1] == Constant("new", TermType.STRING)

    def test_map_put_new_key(self):
        m = MapTerm(((Constant("a", TermType.STRING), Constant(1, TermType.NUMBER)),))
        result = eval_funcall(FunCall("fn:map:put", (m, Constant("b", TermType.STRING), Constant(2, TermType.NUMBER))))
        assert isinstance(result, MapTerm)
        assert len(result.entries) == 2

    def test_map_size(self):
        m = MapTerm((
            (Constant("a", TermType.STRING), Constant(1, TermType.NUMBER)),
            (Constant("b", TermType.STRING), Constant(2, TermType.NUMBER)),
        ))
        result = eval_funcall(FunCall("fn:map:size", (m,)))
        assert result == Constant(2, TermType.NUMBER)


# ── Struct functions ────────────────────────────────────────────────


class TestStruct:
    def test_struct_get(self):
        s = StructTerm((("name", Constant("alice", TermType.STRING)), ("age", Constant(30, TermType.NUMBER))))
        result = eval_funcall(FunCall("fn:struct:get", (s, Constant("name", TermType.STRING))))
        assert result == Constant("alice", TermType.STRING)

    def test_struct_get_missing_field(self):
        s = StructTerm((("name", Constant("alice", TermType.STRING)),))
        result = eval_funcall(FunCall("fn:struct:get", (s, Constant("missing", TermType.STRING))))
        assert result is None

    def test_struct_set(self):
        s = StructTerm((("name", Constant("alice", TermType.STRING)), ("age", Constant(30, TermType.NUMBER))))
        result = eval_funcall(FunCall("fn:struct:set", (s, Constant("age", TermType.STRING), Constant(31, TermType.NUMBER))))
        assert isinstance(result, StructTerm)
        # age updated
        assert result.fields[1] == ("age", Constant(31, TermType.NUMBER))


# ── fn:type ─────────────────────────────────────────────────────────


class TestTypeFunction:
    def test_type_number(self):
        result = eval_funcall(FunCall("fn:type", (Constant(42, TermType.NUMBER),)))
        assert result == Constant("number", TermType.STRING)

    def test_type_string(self):
        result = eval_funcall(FunCall("fn:type", (Constant("hello", TermType.STRING),)))
        assert result == Constant("string", TermType.STRING)

    def test_type_float(self):
        result = eval_funcall(FunCall("fn:type", (Constant(3.14, TermType.FLOAT),)))
        assert result == Constant("float", TermType.STRING)

    def test_type_list(self):
        result = eval_funcall(FunCall("fn:type", (ListTerm(()),)))
        assert result == Constant("list", TermType.STRING)

    def test_type_map(self):
        result = eval_funcall(FunCall("fn:type", (MapTerm(()),)))
        assert result == Constant("map", TermType.STRING)

    def test_type_struct(self):
        result = eval_funcall(FunCall("fn:type", (StructTerm(()),)))
        assert result == Constant("struct", TermType.STRING)


# ── Type predicates ─────────────────────────────────────────────────


class TestTypePredicates:
    def test_is_number_true(self):
        assert eval_builtin_pred(":is_number", [Constant(42, TermType.NUMBER)]) is True

    def test_is_number_float(self):
        assert eval_builtin_pred(":is_number", [Constant(3.14, TermType.FLOAT)]) is True

    def test_is_number_string(self):
        assert eval_builtin_pred(":is_number", [Constant("x", TermType.STRING)]) is False

    def test_is_string_true(self):
        assert eval_builtin_pred(":is_string", [Constant("hello", TermType.STRING)]) is True

    def test_is_string_false(self):
        assert eval_builtin_pred(":is_string", [Constant(42, TermType.NUMBER)]) is False

    def test_is_list_true(self):
        assert eval_builtin_pred(":is_list", [ListTerm(())]) is True

    def test_is_list_false(self):
        assert eval_builtin_pred(":is_list", [Constant(1, TermType.NUMBER)]) is False

    def test_is_map_true(self):
        assert eval_builtin_pred(":is_map", [MapTerm(())]) is True

    def test_is_map_false(self):
        assert eval_builtin_pred(":is_map", [Constant(1, TermType.NUMBER)]) is False

    def test_is_struct_true(self):
        assert eval_builtin_pred(":is_struct", [StructTerm(())]) is True

    def test_is_struct_false(self):
        assert eval_builtin_pred(":is_struct", [Constant(1, TermType.NUMBER)]) is False


# ── Predicate edge cases ────────────────────────────────────────────


class TestPredicateEdgeCases:
    def test_starts_with_wrong_args(self):
        assert eval_builtin_pred(":string:starts_with", [Constant("hello", TermType.STRING)]) is False

    def test_contains_non_constant(self):
        assert eval_builtin_pred(":string:contains", [ListTerm(()), Constant("x", TermType.STRING)]) is False

    def test_list_member_wrong_type(self):
        assert eval_builtin_pred(":list:member", [Constant(1, TermType.NUMBER), Constant(1, TermType.NUMBER)]) is False

    def test_list_empty_wrong_type(self):
        assert eval_builtin_pred(":list:empty", [Constant(1, TermType.NUMBER)]) is False

    def test_is_number_no_args(self):
        assert eval_builtin_pred(":is_number", []) is False

    def test_is_string_non_constant(self):
        assert eval_builtin_pred(":is_string", [ListTerm(())]) is False


# ── Unknown ─────────────────────────────────────────────────────────


class TestUnknown:
    def test_unknown_funcall(self):
        result = eval_funcall(FunCall("fn:nonexistent", ()))
        assert result is None

    def test_unknown_pred(self):
        result = eval_builtin_pred(":nonexistent", [])
        assert result is False
