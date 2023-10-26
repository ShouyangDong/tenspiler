from abc import abstractmethod
from enum import Enum
from inspect import isclass

from llvmlite.binding import TypeRef, ValueRef
from collections import Counter
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Set,
    TypeVar,
    Union,
    Optional,
    Tuple,
    _GenericAlias,
    get_args,
    get_origin,
)
from metalift.types import (Type, FnT, PointerT, TupleT)


class PrintMode(Enum):
    SMT = 0
    Rosette = 1


T = TypeVar("T")


class Expr:
    def __init__(
        self,
        type: typing.Type["NewObject"],
        args: Any,
        metadata: Dict[str, Any] = {}
    ) -> None:
        self.args = args
        self.type = type
        self.metadata = metadata

    # TODO: move into per-type implementations
    def mapArgs(self, f: Callable[["Expr"], "Expr"]) -> "Expr":
        if isinstance(self, NewObject):
            from copy import deepcopy
            new_object = deepcopy(self)
            new_object.src = self.src.mapArgs(f)
            return new_object
        if isinstance(self, Var):
            # TODO(jie)
            return Var(typing.cast(str, f(self.args[0])), self.type)
        elif isinstance(self, Lit):
            return Lit(typing.cast(Union[bool, int, str], f(self.args[0])), self.type)
        elif isinstance(self, Add):
            return Add(*[f(a) for a in self.args])
        elif isinstance(self, Sub):
            return Sub(*[f(a) for a in self.args])
        elif isinstance(self, Mul):
            return Mul(*[f(a) for a in self.args])
        elif isinstance(self, Implies):
            return Implies(*[f(a) for a in self.args])
        elif isinstance(self, And):
            return And(*[f(a) for a in self.args])
        elif isinstance(self, Or):
            return Or(*[f(a) for a in self.args])
        elif isinstance(self, Not):
            return Not(*[f(a) for a in self.args])
        elif isinstance(self, Eq):
            return Eq(*[f(a) for a in self.args])
        elif isinstance(self, Ge):
            return Ge(*[f(a) for a in self.args])
        elif isinstance(self, Gt):
            return Gt(*[f(a) for a in self.args])
        elif isinstance(self, Le):
            return Le(*[f(a) for a in self.args])
        elif isinstance(self, Lt):
            return Lt(*[f(a) for a in self.args])
        elif isinstance(self, Ite):
            return Ite(*[f(a) for a in self.args])
        elif isinstance(self, Tuple):
            return Tuple(*[f(a) for a in self.args])
        elif isinstance(self, Let):
            return Let(*[f(a) for a in self.args])
        elif isinstance(self, Lambda):
            return Lambda(self.type.args[0], *[f(a) for a in self.args])
        elif isinstance(self, Choose):
            return Choose(*[f(a) for a in self.args])
        elif isinstance(self, TupleGet):
            return TupleGet(f(self.args[0]), *[f(a) for a in self.args[1:]])
        elif isinstance(self, Call):
            return Call(
                typing.cast(str, f(self.args[0])),
                self.type,
                *[f(a) for a in self.args[1:]],
            )
        elif isinstance(self, CallValue):
            return CallValue(f(self.args[0]), *[f(a) for a in self.args[1:]])
        else:
            raise Exception("NYI: %s" % self)

    def chooseArbitrarily(self) -> "Expr":
        return self.mapArgs(
            lambda x: x.chooseArbitrarily() if isinstance(x, Expr) else x
        )

    @staticmethod
    def findCommonExprs(
        e: "Expr", cnts: List[Tuple["Expr", int]]
    ) -> List[Tuple["Expr", int]]:
        def expr_index_in_cnts(e: Expr):
            for i, (existing_expr, _) in enumerate(cnts):
                if Expr.__eq__(e, existing_expr):
                    return i
            return -1

        expr_index = expr_index_in_cnts(e)
        if expr_index == -1:
            cnts.append((e, 1))
            for i in range(len(e.args)):
                if isinstance(e.args[i], Expr):
                    cnts = Expr.findCommonExprs(e.args[i], cnts)
        else:
            _, cnt = cnts[expr_index]
            cnts[expr_index] = (e, cnt + 1)
        return cnts

    @staticmethod
    def replaceExprs(
        e: Union[bool, "Expr", ValueRef, int, str],
        commonExprs: typing.List[Union["Expr", Any]],
        mode: PrintMode,
        skipTop: bool = False,
    ) -> Union["Expr", ValueRef]:
        # skipTop is used to ignore the top-level match when simplifying a common expr
        if all([not Expr.__eq__(e, expr) for expr in commonExprs]) or skipTop:
            if isinstance(e, Expr):
                newArgs = [Expr.replaceExprs(arg, commonExprs, mode) for arg in e.args]
                if isinstance(e, NewObject):
                    return Expr.replaceExprs(e.src, commonExprs, mode)
                if isinstance(e, Var):
                    return Var(typing.cast(str, newArgs[0]), e.type)
                elif isinstance(e, Lit):
                    return Lit(typing.cast(Union[bool, int, str], newArgs[0]), e.type)
                elif isinstance(e, Ge):
                    return Ge(*newArgs)
                elif isinstance(e, Gt):
                    return Gt(*newArgs)
                elif isinstance(e, Le):
                    return Le(*newArgs)
                elif isinstance(e, Lt):
                    return Lt(*newArgs)
                elif isinstance(e, Eq):
                    return Eq(*newArgs)
                elif isinstance(e, Ite):
                    return Ite(*newArgs)
                elif isinstance(e, And):
                    return And(*newArgs)
                elif isinstance(e, Or):
                    return Or(*newArgs)
                elif isinstance(e, Not):
                    return Not(*newArgs)
                elif isinstance(e, Implies):
                    return Implies(*newArgs)
                elif isinstance(e, Add):
                    return Add(*newArgs)
                elif isinstance(e, Sub):
                    return Sub(*newArgs)
                elif isinstance(e, Mul):
                    return Mul(*newArgs)
                elif isinstance(e, Call):
                    return Call(typing.cast(str, newArgs[0]), e.type, *newArgs[1:])
                elif isinstance(e, Choose):
                    return Choose(*newArgs)
                elif isinstance(e, Tuple):
                    return Tuple(*newArgs)
                elif isinstance(e, TupleGet):
                    return TupleGet(*newArgs)
                elif isinstance(e, Let):
                    return Let(*newArgs)
                elif isinstance(e, Lambda):
                    return Lambda(e.type.args[0], *newArgs)
                else:
                    raise Exception("NYI: %s" % e)
            else:
                return e  # ValueRef or TypeRef
        else:
            assert isinstance(e, Expr)
            if mode == PrintMode.Rosette:
                return Var("(v%d)" % commonExprs.index(e), e.type)
            else:
                return Var("v%d" % commonExprs.index(e), e.type)

    def __repr__(self) -> str:
        fn: Callable[[Union[ValueRef, Var]], Any] = (
            lambda a: a.name if isinstance(a, ValueRef) and a.name != "" else str(a)
        )

        return (
            f"({type(self).__name__}:{get_type_str(self.type)} "
            f'{" ".join(fn(a) for a in self.args)})'
        )

    def codegen(self) -> str:
        fn: Callable[[Union[ValueRef, Var]], Any] = (
            lambda a: a.name if isinstance(a, ValueRef) and a.name != "" else str(a)
        )
        return (
            f"({type(self).__name__}:{get_type_str(self.type)} "
            f'{" ".join(str(a.codegen()) if isinstance(a, Expr) else fn(a) for a in self.args)})'
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, NewObject) and isinstance(other, NewObject):
            return Expr.__eq__(self.src, other.src)
        elif isinstance(other, Expr):
            if (
                type(self) != type(other)
                # or self.type.erase() #TODO: add them back
                # != other.type.erase()
                or len(self.args) != len(other.args)
            ):
                return False
            else:
                for a1, a2 in zip(self.args, other.args):
                    if isinstance(a1, Type) and isinstance(a2, Type):
                        if a1 != a2:
                            return False
                        continue
                    elif isinstance(a1, Expr) and isinstance(a2, Expr):
                        if not Expr.__eq__(a1, a2):
                            return False
                        continue
                    elif a1 != a2:
                        return False
                return True
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted({"type": self.type, "args": tuple(self.args)})))

    def toSMT(self) -> str:
        raise NotImplementedError

    def accept(self, v: "Visitor[T]") -> T:
        raise NotImplementedError("not implemented")

    @staticmethod
    def toSMTSimple(e: "Expr", name: str) -> str:
        value = name
        return (
            "("
            + value
            + " "
            + " ".join(
                [
                    a.name
                    if isinstance(a, ValueRef) and a.name != ""
                    else a.toSMT()
                    if isinstance(a, Expr)
                    else str(a)
                    for a in e.args
                ]
            )
            + ")"
        )

    listFns = {
        "list_get": "list-ref-noerr",
        "list_append": "list-append",
        "list_empty": "list-empty",
        "list_tail": "list-tail-noerr",
        "list_length": "length",
        "list_take": "list-take-noerr",
        "list_prepend": "list-prepend",
        "list_eq": "equal?",
        "list_concat": "list-concat",
    }

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        raise NotImplementedError()

    @staticmethod
    def toRosetteSimple(e: "Expr", value: str) -> str:
        retStr = "(" + value + " "
        for a in e.args:
            if isinstance(a, ValueRef) and a.name != "":
                retStr += "%s" % (a.name) + " "
            else:
                strExp = a.toRosette() if isinstance(a, Expr) else str(a)
                if (strExp) in Expr.listFns.keys() and "list_empty" in (strExp):
                    retStr += "(" + Expr.listFns[strExp] + ")" + " "
                elif (strExp) in Expr.listFns.keys():
                    retStr += Expr.listFns[strExp]
                else:
                    retStr += strExp + " "
        retStr += ")"
        return retStr

    def simplify(self) -> "Expr":
        self = self.mapArgs(lambda a: a.simplify() if isinstance(a, Expr) else a)
        if isinstance(self, And):
            filtered_args: typing.List[Expr] = []
            for arg in self.args:
                if isinstance(arg, Expr) and isinstance(arg, Lit):
                    if arg.args[0] == False:
                        return BoolLit(False)
                else:
                    filtered_args.append(arg)

            if len(filtered_args) == 0:
                return BoolLit(True)
            elif len(filtered_args) == 1:
                return filtered_args[0]
            else:
                return And(*filtered_args)
        else:
            return self

    def countVariableUses(self, into: Dict[str, int]) -> None:
        if isinstance(self, Var):
            if not (self.args[0] in into):
                into[self.args[0]] = 0
            into[self.args[0]] += 1
        else:
            for a in self.args:
                if isinstance(a, Expr):
                    a.countVariableUses(into)

    def collectKnowledge(
        self,
        constrained_elsewhere: typing.Set[str],
        into: Dict[str, "Expr"],
        conflicts: Dict[str, bool],
    ) -> None:
        if isinstance(self, Eq):
            if isinstance(self.args[0], Var) and (
                not self.args[0].args[0] in constrained_elsewhere
            ):
                if self.args[0].args[0] in into or self.args[0].args[0] in conflicts:
                    conflicts[self.args[0].args[0]] = True
                    del into[self.args[0].args[0]]
                else:
                    into[self.args[0].args[0]] = self.args[1]
            elif isinstance(self.args[1], Var) and (
                not self.args[1].args[0] in constrained_elsewhere
            ):
                if self.args[1].args[0] in into or self.args[1].args[0] in conflicts:
                    conflicts[self.args[1].args[0]] = True
                    del into[self.args[1].args[0]]
                else:
                    into[self.args[1].args[0]] = self.args[0]
        elif isinstance(self, And):
            for a in self.args:
                if isinstance(a, Expr):
                    a.collectKnowledge(constrained_elsewhere, into, conflicts)
        else:
            return

    def rewrite(self, mappings: Dict[str, "Expr"]) -> "Expr":
        if isinstance(self, Var):
            if self.args[0] in mappings:
                return mappings[self.args[0]]
            else:
                return self
        else:
            return self.mapArgs(
                lambda a: a.rewrite(mappings) if isinstance(a, Expr) else a
            )

    def optimizeUselessEquality(
        self, counts: Dict[str, int], new_vars: typing.Set["Var"]
    ) -> "Expr":
        if isinstance(self, Eq):
            replacement_var = Var("useless_equality_%d" % len(new_vars), BoolObject)
            if isinstance(self.args[0], Var) and counts[self.args[0].args[0]] == 1:
                new_vars.add(replacement_var)
                return replacement_var
            elif isinstance(self.args[1], Var) and counts[self.args[1].args[0]] == 1:
                new_vars.add(replacement_var)
                return replacement_var
            elif isinstance(self.args[0], Var) and isinstance(self.args[1], Var):
                if self.args[0].args[0] == self.args[1].args[0]:
                    return BoolLit(True)
        elif isinstance(self, Implies):
            local_counts: Dict[str, int] = {}
            self.countVariableUses(local_counts)

            constrained_elsewhere = set(counts.keys())
            for key in local_counts.keys():
                if local_counts[key] == counts[key]:
                    constrained_elsewhere.remove(key)
            rewrites: Dict[str, "Expr"] = {}
            self.args[0].collectKnowledge(constrained_elsewhere, rewrites, {})

            counts_rhs: Dict[str, int] = {}
            self.args[1].countVariableUses(counts_rhs)

            for rewrite_var in list(rewrites.keys()):
                if not (isinstance(rewrites[rewrite_var], Var)):
                    if rewrite_var in counts_rhs and counts_rhs[rewrite_var] > 1:
                        del rewrites[rewrite_var]

            self = self.rewrite(rewrites)

        return self.mapArgs(
            lambda a: a.optimizeUselessEquality(counts, new_vars)
            if isinstance(a, Expr)
            else a
        ).simplify()

    # convenience methods
    def __add__(self, other: "Expr") -> "Add":
        if isinstance(self, Add):
            return Add(*self.args, other)
        else:
            return Add(self, other)

    def __sub__(self, other: "Expr") -> "Sub":
        if isinstance(self, Sub):
            return Sub(*self.args, other)
        else:
            return Sub(self, other)

    def __mul__(self, other: "Expr") -> "Mul":
        if isinstance(self, Mul):
            return Mul(*self.args, other)
        else:
            return Mul(self, other)


### START OF IR OBJECTS
ObjectContainedT = Union[typing.Type["NewObject"], _GenericAlias]

def get_type_str(type: Union[Type, typing.Type["NewObject"]]):
    if isinstance(type, Type):
        return str(type)
    else:
        return type.cls_str(get_args(type))

def toRosetteType(t: typing.Type["NewObject"]) -> str:
    if t == IntObject:
        return "integer?"
    elif t == BoolObject:
        return "boolean?"
    else:
        raise Exception("NYI: %s" % t)

# TODO(jie): fix the type in the function signature
def parse_type_ref_to_obj(t: TypeRef) -> typing.Type["NewObject"]:
    if is_new_object_type(t):
        return t
    ty_str = str(t)
    if ty_str in {"i32", "i32*"}:
        return IntObject
    elif ty_str == "i1":
        return BoolObject
    elif ty_str in {"%struct.list*", "%struct.list**", '%"class.std::__1::vector"*'}:
        # TODO colin: add generic type support
        # TODO jie: retire struct.list and use STL?
        return ListObject[IntObject]
    elif ty_str in {"%struct.set*"}:
        # TODO jie: how to support different contained types
        return SetObject[IntObject]
    else:
        raise Exception(f"no type defined for {ty_str}")


def create_object(
    object_type: typing.Type["NewObject"], value: Optional[Union[bool, str, Expr]]
) -> "NewObject":
    if isinstance(object_type, _GenericAlias):
        object_cls = get_origin(object_type)
        contained_types = get_args(object_type)
        return object_cls(*contained_types, value)
    else:
        return object_type(value)

def get_object_sources(objects: List["NewObject"]) -> List[Expr]:
    eturn [obj.src for obj in objects]

def is_new_object_type(ty: ObjectContainedT) -> bool:
    return isclass(ty) and issubclass(ty, NewObject)

def choose(*objects: "NewObject") -> "NewObject":
    if len(objects) == 0:
        raise Exception("Must have at least 1 object to choose from!")
    # Assume that all objects passed in will have the same type, even if not, the call to Choose
    # will handle exception throwing for us.
    object_type = objects[0].type
    choose_expr = Choose(*get_object_sources(objects))
    return create_object(object_type, choose_expr)

def ite(cond: "BoolObject", then_object: "NewObject", else_object: "NewObject") -> "NewObject":
    ite_type = then_object.type
    ite_expr = Ite(cond.src, then_object.src, else_object.src)
    return create_object(ite_type, ite_expr)

def implies(e1: "BoolObject", e2: "BoolObject") -> "BoolObject":
    return BoolObject(Implies(e1.src, e2.src))

def call(fn_name: str, return_type: typing.Type["NewObject"], *object_args: "NewObject") -> "NewObject":
    call_expr = Call(fn_name, return_type, *get_object_sources(object_args))
    return create_object(return_type, call_expr)

class NewObject:
    src: Expr

    def __init__(self, src: Expr) -> None:
        self.src = src
        self.__class__.__hash__ = NewObject.__hash__

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return self.src.toRosette(writeChoicesTo)

    def var_name(self) -> str:
        if not isinstance(self.src, Var):
            raise Exception("source is not a variable")
        return typing.cast(Var, self.src).name()

    def __repr__(self) -> str:
        return repr(self.src)

    def toSMT(self) -> str:
        return self.src.toSMT()

    def codegen(self) -> str:
        return self.src.codegen()

    def __hash__(self) -> int:
        return hash(self.src)

    @staticmethod
    def toSMTType(type_args: Tuple[ObjectContainedT] = ()) -> str:
        raise NotImplementedError()

    @staticmethod
    def cls_str(type_args: Tuple[ObjectContainedT] = ()) -> str:
        raise NotImplementedError()


class BoolObject(NewObject):
    def __init__(self, value: Optional[Union[bool, str, Expr]] = None) -> None:
        if value is None:
            src = Var("v", BoolObject)
        elif isinstance(value, bool):
            src = BoolLit(value)
        elif isinstance(value, Expr):
            if value.type != BoolObject:
                raise TypeError(f"Cannot create BoolObject from {value.type}")
            src = value
        elif isinstance(value, str):
            src = Var(value, BoolObject)
        else:
            raise TypeError(f"Cannot create Bool from {value}")

        NewObject.__init__(self, src)

    @property
    def type(self) -> typing.Type["BoolObject"]:
        return BoolObject

    # python doesn't have hooks for and, or, not
    def And(self, *args: "BoolObject") -> "BoolObject":
        if len(args) == 0:
            raise Exception(f"Arg list must be non-empty: {args}")
        return BoolObject(And(*get_object_sources([self, *args])))

    def Or(self, *args: "BoolObject") -> "BoolObject":
        if len(args) == 0:
            raise Exception(f"Arg list must be non-empty: {args}")
        return BoolObject(Or(*get_object_sources([self, *args])))

    def Not(self) -> "BoolObject":
        return BoolObject(Not(self.src))

    def __eq__(self, other: Union["BoolObject", bool]) -> "BoolObject":
        if isinstance(other, bool):
            other = BoolObject(other)
        return BoolObject(Eq(self.src, other.src))

    def __ne__(self, other: Union["BoolObject", bool]) -> "BoolObject":
        if isinstance(other, bool):
            other = BoolObject(other)
        return BoolObject(Not(Eq(self.src, other.src)))

    def __repr__(self):
        return f"{self.src}"

    @staticmethod
    def toSMTType(type_args: Tuple[ObjectContainedT] = ()) -> str:
        return "Bool"

    @staticmethod
    def cls_str(type_args: Tuple[ObjectContainedT] = ()) -> str:
        return "Bool"


class IntObject(NewObject):
    def __init__(self, value: Optional[Union[int, str, Expr]] = None) -> None:
        if value is None:
            src = Var("v", IntObject)
        elif isinstance(value, int):
            src = IntLit(value)
        elif isinstance(value, Expr):
            if value.type != IntObject:
                raise TypeError(f"Cannot create IntObject from {value.type}")
            src = value
        elif isinstance(value, str):
            src = Var(value, IntObject)
        else:
            raise TypeError(f"Cannot create Int from {value}")

        NewObject.__init__(self, src)

    @property
    def type(self) -> typing.Type["IntObject"]:
        return IntObject

    def binary_op(
        self, other: Union["IntObject", int], op: Callable[[Expr, Expr], Expr]
    ) -> "IntObject":
        if isinstance(other, int):
            other = IntObject(other)
        if isinstance(other, IntObject):
            return IntObject(op(self.src, other.src))
        raise TypeError(f"Cannot call {op} on IntObject {self} and {other}")

    # arithmetic operators
    def __add__(self, other: Union["IntObject", int]) -> "IntObject":
        return self.binary_op(other, Add)

    def __sub__(self, other: Union["IntObject", int]) -> "IntObject":
        return self.binary_op(other, Sub)

    def __mul__(self, other: Union["IntObject", int]) -> "IntObject":
        return self.binary_op(other, Mul)

    # div not supported yet

    def __radd__(self, other: Union["IntObject", int]) -> "IntObject":
        if isinstance(other, int):
            return IntObject(other) + self
        else:
            return other + self

    def __rsub__(self, other: Union["IntObject", int]) -> "IntObject":
        if isinstance(other, int):
            return IntObject(other) - self
        else:
            return other - self

    def __rmul__(self, other: Union["IntObject", int]) -> "IntObject":
        if isinstance(other, int):
            return IntObject(other) * self
        else:
            return other * self

    # logical comparison operators
    def __eq__(self, other: Union["IntObject", int]) -> BoolObject:
        if isinstance(other, int):
            other = IntObject(other)
        return BoolObject(Eq(self.src, other.src))

    def __ne__(self, other: Union["IntObject", int]) -> BoolObject:
        if isinstance(other, int):
            other = IntObject(other)
        return BoolObject(Not(Eq(self.src, other.src)))

    def __ge__(self, other: Union["IntObject", int]) -> BoolObject:
        if isinstance(other, int):
            other = IntObject(other)
        return BoolObject(Ge(self.src, other.src))

    def __gt__(self, other: Union["IntObject", int]) -> BoolObject:
        if isinstance(other, int):
            other = IntObject(other)
        return BoolObject(Gt(self.src, other.src))

    def __lt__(self, other: Union["IntObject", int]) -> BoolObject:
        if isinstance(other, int):
            other = IntObject(other)
        return BoolObject(Lt(self.src, other.src))

    def __le__(self, other: Union["IntObject", int]) -> BoolObject:
        if isinstance(other, int):
            other = IntObject(other)
        return BoolObject(Le(self.src, other.src))


    @staticmethod
    def toSMTType(type_args: Tuple[ObjectContainedT] = ()) -> str:
        return "Int"

    @staticmethod
    def cls_str(type_args: Tuple[ObjectContainedT] = ()) -> str:
        return "Int"


T = TypeVar("T", bound=NewObject)

class ListObject(Generic[T], NewObject):
    def __init__(
        self,
        containedT: ObjectContainedT = IntObject,
        value: Optional[Union[Expr, str]] = None,
    ) -> None:
        full_type = ListObject[containedT]
        if value is None:  # a symbolic variable
            src = Var("v", full_type)
        elif isinstance(value, Expr):
            src = value
        elif isinstance(value, str):
            src = Var(value, full_type)
        else:
            raise TypeError(f"Cannot create ListObject from {value}")
        self.containedT = containedT
        NewObject.__init__(self, src)

    @property
    def type(self) -> typing.Type["ListObject"]:
        return ListObject[self.containedT]

    @staticmethod
    def empty(containedT: ObjectContainedT) -> "ListObject":
        return ListObject(containedT, Call("list_empty", ListObject[containedT]))

    def __len__(self) -> int:
        raise NotImplementedError("len must return an int, call len() instead")

    def len(self) -> IntObject:
        return IntObject(Call("list_length", IntObject, self.src))

    def __getitem__(
        self, index: Union[IntObject, int, slice]
    ) -> Union[IntObject, "ListObject[IntObject]"]:
        if isinstance(index, int):
            index = IntObject(index)
        if isinstance(index, slice):
            (start, stop, step) = (index.start, index.stop, index.step)
            if stop is None and step is None:
                if isinstance(start, int):
                    start = IntObject(start)
                return ListObject(
                    self.containedT,
                    Call("list_tail", ListObject[self.containedT], self.src, start.src)
                )
            elif start is None and step is None:
                if isinstance(stop, int):
                    stop = IntObject(stop)
                return ListObject(
                    self.containedT,
                    Call("list_head", ListObject[self.containedT], self.src, stop.src)
                )
            else:
                raise NotImplementedError(f"Slices with both start and stop indices specified are not implemented: {index}")

        if is_new_object_type(self.containedT):  # non generic type
            return self.containedT(Call("list_get", self.containedT, self.src, index.src))
        elif isinstance(self.containedT, _GenericAlias):  # generic type
            subcontainedT = typing.get_args(self.containedT)[0]
            return self.containedT(
                subcontainedT, Call("list_get", self.containedT, self.src, index.src)
            )
        else:
            raise NotImplementedError(
                f"Cannot get item from list containing type {self.containedT}"
            )

    def __setitem__(self, index: Union[IntObject, int], value: NewObject) -> None:
        if isinstance(index, int):
            index = IntObject(index)
        if value.type != self.containedT:
            raise TypeError(
                f"Trying to set list element of type: {value.type} to list containing: {self.containedT}"
            )
        # TODO(jie): is this working
        self.src = Call("list_set", self.type, self.src, index.src, value.src)

    # in place append
    def append(self, value: NewObject) -> "ListObject":
        if value.type != self.containedT:
            raise TypeError(
                f"Trying to append element of type: {value.type} to list containing: {self.containedT}"
            )

        self.src = Call("list_append", self.type, self.src, value.src)
        return self

    # list concat that returns a new list
    def __add__(self, other: "ListObject") -> "ListObject":
        if self.type != other.type:
            raise TypeError(
                f"can't add lists of different types: {self.type} and {other.type}"
            )
        return ListObject(
            self.containedT, Call("list_concat", self.type, self.src, other.src)
        )

    def __eq__(self, other: "ListObject") -> BoolObject:
        if other is None or self.type != other.type:
            return BoolObject(False)
        else:
            return call("list_eq", BoolObject, self, other)

    def __repr__(self):
        return f"{self.src}"

    @staticmethod
    def cls_str(type_args: Tuple[ObjectContainedT] = ()) -> str:
        contained_type = type_args[0]
        if isinstance(contained_type, _GenericAlias):
            return f"List {get_origin(contained_type).cls_str(get_args(contained_type))}"
        else:
            return f"List {contained_type.cls_str()}"

    @staticmethod
    def toSMTType(type_args: Tuple[ObjectContainedT] = ()) -> str:
        contained_type = type_args[0]
        if isinstance(contained_type, _GenericAlias):
            return f"(MLList {get_origin(contained_type).toSMTType(get_args(contained_type))})"
        else:
            return f"(MLList {contained_type.toSMTType()})"

class SetObject(Generic[T], NewObject):
    def __init__(
        self,
        containedT: Union[type, _GenericAlias] = IntObject,
        value: Optional[Union[Expr, str]] = None,
    ) -> None:
        full_type = SetObject[containedT]
        if value is None:
            src = Var("v", full_type)
        elif isinstance(value, Expr):
            src = value
        elif isinstance(value, str):
            src = Var(value, full_type)
        else:
            raise TypeError(f"Cannot create SetObject from {value}")
        self.containedT = containedT
        NewObject.__init__(self, src)

    @property
    def type(self) -> typing.Type["SetObject"]:
        return SetObject[self.containedT]

    def add(self, value: NewObject) -> "SetObject":
        if value.type != self.containedT:
            raise TypeError(
                f"Trying to add element of type: {value.type} to set containing: {self.containedT}"
            )
        return call("set-insert", self.type, value, self)

    def remove(self, item: NewObject) -> "SetObject":
        if type(item) != self.containedT:
            raise TypeError(f"Trying to remove element of type: {type(item)} from set containing: {self.containedT}")
        singleton_s = SetObject.singleton(item)
        return call("set-minus", self.type, self, singleton_s)

    @staticmethod
    def singleton(item: NewObject) -> "SetObject":
        return call("set-singleton", SetObject[type(item)], item)

    def union(self, s: "SetObject") -> "SetObject":
        if self.type != s.type:
            raise TypeError(f"Can't union two sets with type {self.type} and type {s.type}")
        expr = Call("set-union", self.type, self.src, s.src)
        return SetObject(self.containedT, expr)

    def difference(self, s: "SetObject") -> "SetObject":
        if self.type != s.type:
            raise TypeError(f"Can't take the difference of two sets with type {self.type} and type {s.type}")
        return call("set-minus", self.type, self, s)

    def __contains__(self, value: NewObject) -> BoolObject:
        if value.type != self.containedT:
            return BoolObject(False)
        return call("set-pointer_varsber", BoolObject, self, value)

    def __eq__(self, s: "SetObject") -> BoolObject:
        return call("set-eq", BoolObject, self, s)

    @staticmethod
    def empty(containedT: ObjectContainedT) -> "SetObject":
        return SetObject(containedT, Call("set-create", SetObject[containedT]))

    @staticmethod
    def toSMTType(type_args: Tuple[ObjectContainedT] = ()) -> str:
        return SetObject.cls_str(type_args)

    @staticmethod
    def cls_str(type_args: Tuple[ObjectContainedT] = ()) -> str:
        contained_type = type_args[0]
        if isinstance(contained_type, _GenericAlias):
            return f"(Set {get_origin(contained_type).toSMTType(get_args(contained_type))})"  # this would call List.toSMTType(Int) for instance
        else:
            return f"(Set {contained_type.toSMTType()})"

IntT = TypeVar("IntT")
class TupleObject(Generic[T, IntT], NewObject):
    def __init__(
        self,
        containedT: Union[type, _GenericAlias] = IntObject,
        intT: type = Literal[1],
        value: Optional[Union[Expr, str]] = None,
    ) -> None:
        full_type = TupleObject[containedT, intT]
        if value is None:  # a symbolic variable
            src = Var("v", full_type)
        elif isinstance(value, Expr):
            src = value
        elif isinstance(value, str):
            src = Var(value, full_type)
        else:
            raise TypeError(f"Cannot create TupleObject from {value}")
        self.containedT = containedT
        self.intT = intT
        NewObject.__init__(self, src, full_type)

    def __getitem__(
        self, index: Union[IntObject, int]
    ):  # -> IntObject:  # index can also be slice
        if isinstance(index, int):  # promote to IntObject
            index = IntObject(index)
        if isinstance(index, IntObject):
            if issubclass(self.containedT, NewObject):
                # TODO(jie) create a function to wrap objects around expession
                return self.containedT(Call("tupleGet", self.containedT, self, index))
            else:
                raise Exception(
                    "Only primitive object types inside tuples are supported!"
                )
        if isinstance(index, slice):
            raise Exception("slicing operation not supported on tuples!")

    @property
    def length(self) -> int:
        return get_args(self.intT)[0]

    @staticmethod
    def toSMTType(
        type_args: Tuple[Union[typing.Type["NewObject"], _GenericAlias]]
    ) -> str:
        containedT, intT = type_args
        tuple_length = get_args(intT)[0]
        if isinstance(containedT, _GenericAlias):
            containedT_str = get_origin(containedT).toSMTType(get_args(containedT))
        else:
            containedT_str = containedT.toSMTType()
        return f"(Tuple{tuple_length} {' '.join([containedT_str] * tuple_length)})"  # this would call List.toSMTType(Int) for instance

    # TODO(jie): handle contained type
    @staticmethod
    def cls_str() -> str:
        return "Tuple"

### END OF IR OBJECTS

class Var(Expr):
    def __init__(self, name: str, ty: typing.Type[NewObject]) -> None:
        Expr.__init__(self, ty, [name])

    def name(self) -> str:
        return self.args[0]  # type: ignore

    def __repr__(self) -> str:
        return self.args[0]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return str(self.args[0])

    def toSMT(self) -> str:
        return str(self.args[0])

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Var(self)


# used in defining grammars
class NonTerm(Var):
    currentNum = 0  # current non terminal number

    def __init__(self, t: Type, isStart: bool = False, name: str = "") -> None:
        if name == "":
            name = f"nonTerm{NonTerm.currentNum}"
            NonTerm.currentNum = NonTerm.currentNum + 1
        Var.__init__(self, name, t)
        self.isStart = isStart

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_NonTerm(self)


class Pointer(Expr):
    def __init__(self, val: Expr) -> None:
        Expr.__init__(self, PointerT(val.type), [val])

    @property
    def value(self) -> Expr:
        return self.args[0]  # type: ignore

    def set_value(self, value: Expr) -> None:
        self.args[0] = value


class Lit(Expr):
    def __init__(self, val: Union[bool, int, str], ty: Type) -> None:
        Expr.__init__(self, ty, [val])

    def val(self) -> Union[bool, int, str]:
        return self.args[0]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        if self.type == BoolObject:
            return "true" if self.args[0] else "false"
        else:
            return str(self.args[0])

    def toSMT(self) -> str:
        if self.type == BoolObject:
            return "true" if self.args[0] else "false"
        else:
            return str(self.args[0])

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Lit(self)


class ObjectExpr(Expr):
    def __init__(self, ty: Type) -> None:
        Expr.__init__(self, ty, {})

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        raise Exception("NYI")

    def toSMT(self) -> str:
        raise Exception("NYI")

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Object(self)


def IntLit(val: int) -> Expr:
    return Lit(val, IntObject)


def EnumIntLit(val: int) -> Expr:
    return Lit(val, IntObject)
    #TODO: bring EnumBack
    # return Lit(val, EnumInt())


def BoolLit(val: bool) -> Expr:
    return Lit(val, BoolObject)


class Add(Expr):
    RosetteName = SMTName = "+"

    def __init__(self, *args: Expr) -> None:
        if len(args) < 1:
            raise Exception(f"Arg list must be non-empty: {args}")
        # TODO: add this back
        for arg in args:
            if arg.type != args[0].type:
                # raise Exception(
                #     f"Args types not equal: {arg.type.erase()} and {args[0].type.erase()}"
                # )
                # TODO(jie) add back `erase` once we retire types
                raise Exception(f"Args types not equal: {arg.type} and {args[0].type}")
        Expr.__init__(self, IntObject, args)

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Add(self)


class Sub(Expr):
    RosetteName = SMTName = "-"

    def __init__(self, *args: Expr) -> None:
        if len(args) < 1:
            raise Exception(f"Arg list must be non-empty: {args}")
        for arg in args:
            if get_type_str(arg.type) != get_type_str(args[0].type):
                raise Exception(
                    f"Args types not equal: {get_type_str(arg.type)} and {get_type_str(args[0].type)}"
                )
        Expr.__init__(self, IntObject, args)

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Sub(self)


class Mul(Expr):
    RosetteName = SMTName = "*"

    def __init__(self, *args: Expr) -> None:
        if len(args) < 1:
            raise Exception(f"Arg list must be non-empty: {args}")
        for arg in args:
            if get_type_str(arg.type) != get_type_str(args[0].type):
                raise Exception(
                    f"Args types not equal: {get_type_str(arg.type)} and {get_type_str(args[0].type)}"
                )
        Expr.__init__(self, IntObject, args)

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Mul(self)


class Eq(Expr):
    RosetteName = "equal?"
    SMTName = "="

    def __init__(self, e1: Expr, e2: Expr) -> None:
        # if not (e1.type.erase() == e2.type.erase()): TODO: add them back
        #     raise Exception(
        #         f"Cannot compare values of different types: {e1}: {e1.type.erase()} and {e2}: {e2.type.erase()}"
        #     )
        Expr.__init__(self, BoolObject, [e1, e2])

    def e1(self) -> Expr:
        return self.args[0]  # type: ignore

    def e2(self) -> Expr:
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        cmp_type = self.args[0].type
        # TODO(jie)
        if issubclass(cmp_type, NewObject) and cmp_type.cls_str() == "Set":
            name = "set-eq"
        else:
            name = self.RosetteName
        return Expr.toRosetteSimple(self, name)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Eq(self)


class Lt(Expr):
    RosetteName = SMTName = "<"

    def __init__(self, e1: Expr, e2: Expr) -> None:
        if not (get_type_str(e1.type) == get_type_str(e2.type)):
            raise Exception(
                f"Cannot compare values of different types: {e1}: {get_type_str(e1.type)} and {e2}: {get_type_str(e2.type)}"
            )
        Expr.__init__(self, BoolObject, [e1, e2])

    def e1(self) -> Expr:
        return self.args[0]  # type: ignore

    def e2(self) -> Expr:
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Lt(self)


class Le(Expr):
    RosetteName = SMTName = "<="

    def __init__(self, e1: Expr, e2: Expr) -> None:
        if not (get_type_str(e1.type) == get_type_str(e2.type)):
            raise Exception(
                f"Cannot compare values of different types: {e1}: {get_type_str(e1.type)} and {e2}: {get_type_str(e2.type)}"
            )
        Expr.__init__(self, BoolObject, [e1, e2])

    def e1(self) -> Expr:
        return self.args[0]  # type: ignore

    def e2(self) -> Expr:
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Le(self)


class Gt(Expr):
    RosetteName = SMTName = ">"

    def __init__(self, e1: Expr, e2: Expr) -> None:
        if not (get_type_str(e1.type) == get_type_str(e2.type)):
            raise Exception(
                f"Cannot compare values of different types: {e1}: {get_type_str(e1.type)} and {e2}: {get_type_str(e2.type)}"
            )
        Expr.__init__(self, BoolObject, [e1, e2])

    def e1(self) -> Expr:
        return self.args[0]  # type: ignore

    def e2(self) -> Expr:
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Gt(self)


class Ge(Expr):
    RosetteName = SMTName = ">="

    def __init__(self, e1: Expr, e2: Expr) -> None:
        if not (get_type_str(e1.type) == get_type_str(e2.type)):
            raise Exception(
                f"Cannot compare values of different types: {e1}: {get_type_str(e1.type)} and {e2}: {get_type_str(e2.type)}"
            )
        Expr.__init__(self, BoolObject, [e1, e2])

    def e1(self) -> Expr:
        return self.args[0]  # type: ignore

    def e2(self) -> Expr:
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Ge(self)


class And(Expr):
    RosetteName = "&&"  # racket "and" short circuits!
    SMTName = "and"

    def __init__(self, *args: Expr) -> None:
        if len(args) < 1:
            raise Exception(f"Arg list must be non-empty: {args}")
        if not all(map(lambda e: e.type == BoolObject, args)):
            # TODO(jie) how to check this type?
            raise Exception(f"Cannot apply AND to values of type {args}")
        Expr.__init__(self, BoolObject, args)

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_And(self)


class Or(Expr):
    # XXX we should use || for rosette to avoid short circuiting
    RosetteName = SMTName = "or"

    def __init__(self, *args: Expr) -> None:
        if len(args) < 1:
            raise Exception(f"Arg list must be non-empty: {args}")
        if not all(map(lambda e: e.type == BoolObject, args)):
            raise Exception(
                f"Cannot apply OR to values of type {map(lambda e: e.type, args)}"
            )
        Expr.__init__(self, BoolObject, args)

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Or(self)


class Not(Expr):
    RosetteName = "!"
    SMTName = "not"

    def __init__(self, e: Expr) -> None:
        if e.type != BoolObject:
            raise Exception(f"Cannot apply NOT to value of type {e.type}")
        Expr.__init__(self, BoolObject, [e])

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Not(self)


class Implies(Expr):
    RosetteName = SMTName = "=>"

    def __init__(self, e1: Union[Expr, "MLInst"], e2: Union[Expr, "MLInst"]) -> None:
        if e1.type != BoolObject:  # type: ignore
            raise Exception(f"Cannot apply IMPLIES to value of type {e1.type}")  # type: ignore
        if e2.type != BoolObject:  # type: ignore
            raise Exception(f"Cannot apply IMPLIES to value of type {e2.type}")  # type: ignore
        Expr.__init__(self, BoolObject, [e1, e2])

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Implies(self)


class Ite(Expr):
    RosetteName = "if"
    SMTName = "ite"

    def __init__(self, c: Expr, e1: Expr, e2: Expr) -> None:
        if c.type != BoolObject:
            raise Exception(
                f"ITE condition must be Boolean and not value of type {c.type}"
            )
        if get_type_str(e1.type) != get_type_str(e2.type):
            raise Exception(
                f"TE branches in ITE must have the same type: {e1.type}, {e2.type}"
            )
        Expr.__init__(self, e1.type, [c, e1, e2])

    def c(self) -> Expr:
        return self.args[0]  # type: ignore

    def e1(self) -> Expr:
        return self.args[1]  # type: ignore

    def e2(self) -> Expr:
        return self.args[2]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Ite(self)


class Let(Expr):
    def __init__(self, v: Expr, e: Expr, e2: Expr) -> None:
        Expr.__init__(self, e2.type, [v, e, e2])

    def v(self) -> Expr:
        return self.args[0]  # type: ignore

    def e(self) -> Expr:
        return self.args[1]  # type: ignore

    def e2(self) -> Expr:
        return self.args[2]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        let_expr = (
            self.args[1].name
            if isinstance(self.args[1], ValueRef) and self.args[1].name != ""
            else self.args[1]
            if isinstance(self.args[1], str)
            else self.args[1].toRosette()
        )

        return f"(let ([{self.args[0].toRosette()} {let_expr}]) {self.args[2].toRosette()})"

    def toSMT(self) -> str:
        return "(let ((%s %s)) %s)" % (
            self.args[0].toSMT(),
            self.args[1].toSMT(),
            self.args[2].toSMT(),
        )

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Let(self)


class Call(Expr):
    def __init__(
        self, name: str, returnT: typing.Type[NewObject], *arguments: Expr
    ) -> None:
        Expr.__init__(self, returnT, [name, *arguments])

    def name(self) -> str:
        return self.args[0]  # type: ignore

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[1:]  # type: ignore

    def __repr__(self) -> str:
        fn: Callable[[Union[ValueRef, Var]], Any] = (
            lambda a: a.name if isinstance(a, ValueRef) and a.name != "" else str(a)
        )
        return f"({self.args[0]}:{get_type_str(self.type)} {' '.join(fn(a) for a in self.args[1:])})"

    def codegen(self) -> str:
        fn: Callable[[Union[ValueRef, Var]], Any] = (
            lambda a: a.name if isinstance(a, ValueRef) and a.name != "" else str(a)
        )
        return f"({self.args[0]}:{get_type_str(self.type)} {' '.join(str(a.codegen()) if isinstance(a, Expr) else fn(a) for a in self.args[1:])})"

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        if isinstance(self.args[0], str) or isinstance(self, CallValue):
            if isinstance(self.args[0], str) and (
                self.args[0].startswith("inv") or self.args[0].startswith("ps")
            ):
                callStr = "( " + "%s " % (str(self.args[0]))
                for a in self.args[1:]:
                    callStr += a.toRosette() + " "
                callStr += ")"
                return callStr
            elif isinstance(self.args[0], str) and self.args[0].startswith("list"):
                callStr = (
                    "("
                    + "%s"
                    % (
                        Expr.listFns[self.args[0]]
                        if self.args[0] in Expr.listFns.keys()
                        else self.args[0]
                    )
                    + " "
                )
                for a in self.args[1:]:
                    if isinstance(a, ValueRef) and a.name != "":
                        callStr += "%s " % (a.name)
                    else:
                        callStr += a.toRosette() + " "
                callStr += ")"
                return callStr
            else:
                return (
                    "("
                    + " ".join(
                        [
                            a.name
                            if isinstance(a, ValueRef) and a.name != ""
                            else a
                            if isinstance(a, str)
                            else a.toRosette()
                            for a in self.args
                        ]
                    )
                    + ")"
                )
        else:
            return " ".join(
                [
                    a.name
                    if isinstance(a, ValueRef) and a.name != ""
                    else str(a)
                    if isinstance(a, str)
                    else a.toRosette()
                    for a in self.args
                ]
            )

    def toSMT(self) -> str:
        noParens = isinstance(self, Call) and len(self.args) == 1
        retVal = []

        if self.args[0] == "set-create":
            return f"(as set.empty {self.type.toSMT()})"

        if self.args[0] == "tupleGet":
            argvals = self.args[:-1]
        else:
            argvals = self.args

        for idx, a in enumerate(argvals):
            if isinstance(a, ValueRef) and a.name != "":
                retVal.append(a.name)
            elif (str(a)) == "make-tuple":
                retVal.append("tuple%d" % (len(self.args[idx + 1 :])))
            elif (str(a)) == "tupleGet":
                if len(self.args[idx + 1].args) > 0 and self.args[idx + 1].args[0] == "make-tuple":
                    retVal.append(
                        "tuple%d_get%d"
                        % (
                            len(self.args[idx + 1].args) - 1,
                            self.args[idx + 2].args[0],
                        )
                    )
                elif isinstance(self.args[idx + 1], TupleObject):
                    index_expr = self.args[idx + 2]
                    # TODO(jie): this is not very clean, how to make it better?
                    if isinstance(index_expr, IntObject):
                        index = index_expr.src.args[0]
                    else:
                        index = index_expr.args[0]
                    retVal.append(f"tuple{self.args[idx + 1].length}_get{index}")
                else:
                    # HACK: if function argument is a tuple, count I's in the mangled names of args to get number of elements in tuple
                    freq: typing.Counter[str] = Counter(
                        self.args[idx + 1].args[0].split("_")[1]
                    )
                    retVal.append(
                        "tuple%d_get%d" % (freq["i"], self.args[idx + 2].args[0])
                    )
            elif (str(a)).startswith("set-"):
                retVal.append("set.%s" % (str(a)[4:]))
            elif (str(a)).startswith("map-"):
                retVal.append("map_%s" % (str(a)[4:]))
            elif isinstance(a, str):
                retVal.append(str(a))
            else:
                retVal.append(a.toSMT())

        retT = ("" if noParens else "(") + " ".join(retVal) + ("" if noParens else ")")

        return retT

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Call(self)


class CallValue(Expr):
    def __init__(self, value: Expr, *arguments: Expr) -> None:
        Expr.__init__(self, value.type.args[0], [value, *arguments])

    def value(self) -> Expr:
        return self.args[0]  # type: ignore

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        if isinstance(self.args[0], str) or isinstance(self, CallValue):
            if isinstance(self.args[0], str) and (
                self.args[0].startswith("inv") or self.args[0].startswith("ps")
            ):
                callStr = "( " + "%s " % (str(self.args[0]))
                for a in self.args[1:]:
                    callStr += a.toRosette() + " "
                callStr += ")"
                return callStr
            elif isinstance(self.args[0], str) and self.args[0].startswith("list"):
                callStr = (
                    "("
                    + "%s"
                    % (
                        Expr.listFns[self.args[0]]
                        if self.args[0] in Expr.listFns.keys()
                        else self.args[0]
                    )
                    + " "
                )
                for a in self.args[1:]:
                    if isinstance(a, ValueRef) and a.name != "":
                        callStr += "%s " % (a.name)
                    else:
                        callStr += a.toRosette() + " "
                callStr += ")"
                return callStr
            else:
                return (
                    "("
                    + " ".join(
                        [
                            a.name
                            if isinstance(a, ValueRef) and a.name != ""
                            else a
                            if isinstance(a, str)
                            else a.toRosette()
                            for a in self.args
                        ]
                    )
                    + ")"
                )
        else:
            return " ".join(
                [
                    a.name
                    if isinstance(a, ValueRef) and a.name != ""
                    else str(a)
                    if isinstance(a, str)
                    else a.toRosette()
                    for a in self.args
                ]
            )

    def toSMT(self) -> str:
        noParens = isinstance(self, Call) and len(self.args) == 1
        retVal = []

        if self.args[0] == "set-create":
            return f"(as set.empty {self.type.toSMT()})"

        if self.args[0] == "tupleGet":
            argvals = self.args[:-1]
        else:
            argvals = self.args

        for idx, a in enumerate(argvals):
            if isinstance(a, ValueRef) and a.name != "":
                retVal.append(a.name)
            elif (str(a)) == "make-tuple":
                retVal.append("tuple%d" % (len(self.args[idx + 1 :])))
            elif (str(a)) == "tupleGet":
                if self.args[idx + 1].args[0] == "make-tuple":
                    retVal.append(
                        "tuple%d_get%d"
                        % (
                            len(self.args[idx + 1].args) - 1,
                            self.args[idx + 2].args[0],
                        )
                    )
                else:
                    # HACK: if function argument is a tuple, count I's in the mangled names of args to get number of elements in tuple
                    freq: typing.Counter[str] = Counter(
                        self.args[idx + 1].args[0].split("_")[1]
                    )
                    retVal.append(
                        "tuple%d_get%d" % (freq["i"], self.args[idx + 2].args[0])
                    )
            elif (str(a)).startswith("set-"):
                retVal.append("set.%s" % (str(a)[4:]))
            elif (str(a)).startswith("map-"):
                retVal.append("map_%s" % (str(a)[4:]))
            elif isinstance(a, str):
                retVal.append(str(a))
            else:
                retVal.append(a.toSMT())

        retT = ("" if noParens else "(") + " ".join(retVal) + ("" if noParens else ")")

        return retT

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_CallValue(self)


class Assert(Expr):
    RosetteName = SMTName = "assert"

    def __init__(self, e: Expr) -> None:
        Expr.__init__(self, BoolObject, [e])

    def e(self) -> Expr:
        return self.args[0]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return Expr.toRosetteSimple(self, self.RosetteName)

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Assert(self)


class Constraint(Expr):
    SMTName = "constraint"

    def __init__(self, e: Expr) -> None:
        Expr.__init__(self, BoolObject, [e])

    def e(self) -> Expr:
        return self.args[0]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        raise Exception("NYI")

    def toSMT(self) -> str:
        return Expr.toSMTSimple(self, self.SMTName)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Constraint(self)


## tuple functions
class Tuple(Expr):
    def __init__(self, *args: Expr) -> None:
        Expr.__init__(self, TupleT(*[a.type for a in args]), args)

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        # original code was "(make-tuple %s) % " ".join(["%s" % str(arg) for arg in self.args])
        # but arg can be a ValueRef and calling str on it will return both type and name e.g., i32 %arg
        return Call("make-tuple", self.type, *self.args).toRosette()

    def toSMT(self) -> str:
        args = " ".join(
            [
                arg.name if isinstance(arg, ValueRef) else arg.toSMT()
                for arg in self.args
            ]
        )
        return "(tuple%d %s)" % (len(self.args), args)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Tuple(self)


class TupleGet(Expr):
    def __init__(self, t: Expr, i: Expr) -> None:
        Expr.__init__(self, t.type.args[i.args[0]], [t, i])

    def t(self) -> Expr:
        return self.args[0]  # type: ignore

    def i(self) -> Expr:
        return self.args[1]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return "(tupleGet %s)" % " ".join(["%s" % arg.toRosette() for arg in self.args])

    def toSMT(self) -> str:
        # example: generate (tuple2_get0 t)
        return "(tuple%d_get%d %s)" % (
            len(self.args[0].type.args),
            self.args[1].args[0],
            self.args[0].toSMT(),
        )  # args[1] must be an int literal

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_TupleGet(self)


class Axiom(Expr):
    def __init__(self, e: Expr, *vars: Expr) -> None:
        Expr.__init__(self, BoolObject, [e, *vars])

    def e(self) -> Expr:
        return self.args[0]  # type: ignore

    def vars(self) -> typing.List[Expr]:
        return self.args[1:]  # type: ignore

    def toRosette(
        self,
        writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None,
        is_uninterp: bool = True,
    ) -> str:
        return ""  # axioms are only for verification

    def toSMT(self) -> str:
        vs = ["(%s %s)" % (a.args[0], a.type) for a in self.args[1:]]
        return "(assert (forall ( %s ) %s ))" % (" ".join(vs), self.args[0].toSMT())

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Axiom(self)


# the body of a synth-fun
class Synth(Expr):
    def __init__(self, name: str, body: Expr, *args: Expr) -> None:
        Expr.__init__(self, body.type, [name, body, *args])

    def name(self) -> str:
        return self.args[0]  # type: ignore

    def body(self) -> Expr:
        return self.args[1]  # type: ignore

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[2:]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        cnts = Expr.findCommonExprs(self.args[1], [])
        commonExprs = list(
            filter(
                lambda k: isinstance(k, Choose),
                [expr_cnt_tup[0] for expr_cnt_tup in cnts],
            )
        )
        rewritten = Expr.replaceExprs(self.args[1], commonExprs, PrintMode.Rosette)

        # rewrite common exprs to use each other
        commonExprs = [
            Expr.replaceExprs(e, commonExprs, PrintMode.Rosette, skipTop=True)
            for e in commonExprs
        ]

        args = " ".join(
            "%s" % a.name
            if isinstance(a, ValueRef)
            else str(a)
            if isinstance(a, str)
            else a.toRosette()
            for a in self.args[2:]
        )

        defs = "[rv (choose %s)]\n" % rewritten.toRosette()

        if writeChoicesTo != None:
            for i, e in enumerate(commonExprs):
                writeChoicesTo[f"v{i}"] = e  # type: ignore

        defs = defs + "\n".join(
            "%s %s)]" % ("[v%d (choose" % i, e.toRosette())
            for i, e in enumerate(commonExprs)
        )

        return "(define-grammar (%s_gram %s)\n %s\n)" % (self.args[0], args, defs)

    def toSMT(self) -> str:
        cnts = Expr.findCommonExprs(self.args[1], {})
        commonExprs = list(
            filter(
                lambda k: isinstance(k, Choose),
                cnts.keys(),
            )
        )
        rewritten = Expr.replaceExprs(self.args[1], commonExprs, PrintMode.SMT)

        # rewrite common exprs to use each other
        commonExprs = [
            Expr.replaceExprs(e, commonExprs, PrintMode.SMT, skipTop=True)
            for e in commonExprs
        ]

        decls = "((rv %s) %s)" % (
            self.type.toSMT(),
            " ".join(
                "(%s %s)" % ("v%d" % i, parse_type_ref_to_obj(e.type).toSMT())
                for i, e in enumerate(commonExprs)
            ),
        )
        defs = "(rv %s %s)\n" % (
            self.type.toSMT(),
            rewritten.toSMT()
            if isinstance(rewritten, Choose)
            else "(%s)" % rewritten.toSMT(),
        )
        defs = defs + "\n".join(
            "(%s %s %s)"
            % (
                "v%d" % i,
                parse_type_ref_to_obj(e.type).toSMT(),
                e.toSMT() if isinstance(e, Choose) else f"({e.toSMT()})",
            )
            for i, e in enumerate(commonExprs)
        )

        body = decls + "\n" + "(" + defs + ")"

        declarations = []
        for a in self.args[2:]:
            if isinstance(a, ValueRef):
                declarations.append((a.name, a.type))
            else:
                declarations.append((a.args[0], a.type))

        args = " ".join("(%s %s)" % (d[0], d[1].toSMT()) for d in declarations)
        return "(synth-fun %s (%s) %s\n%s)" % (
            self.args[0],
            args,
            self.type.toSMT(),
            body,
        )

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Synth(self)


class Choose(Expr):
    def __init__(self, *args: Expr) -> None:
        if not all(a.type == args[0].type for a in args):
            raise Exception(
                "Choose args are of different types: %s"
                % " ".join(str(a.type) for a in args)
            )
        Expr.__init__(self, args[0].type, args)

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return " ".join(
            [
                a.name
                if isinstance(a, ValueRef) and a.name != ""
                else str(a)
                if isinstance(a, str)
                else a.toRosette()
                for a in self.args
            ]
        )

    def toSMT(self) -> str:
        retVal = []

        for a in self.args:
            if isinstance(a, ValueRef) and a.name != "":
                retVal.append(a.name)
            elif isinstance(a, str):
                retVal.append(str(a))
            else:
                retVal.append(a.toSMT())

        retT = "(" + " ".join(retVal) + ")"

        return retT

    def chooseArbitrarily(self) -> "Expr":
        return self.args[0]  # type: ignore

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Choose(self)


class FnDeclRecursive(Expr):
    def __init__(
        self,
        name: str,
        returnT: typing.Type[NewObject],
        body: Union[Expr, str],
        *args: Expr,
    ) -> None:
        Expr.__init__(self, FnT(returnT, *[a.type for a in args]), [name, body, *args])

    def name(self) -> str:
        return self.args[0]  # type: ignore

    def returnT(self) -> typing.Type[NewObject]:
        return self.type.args[0]

    def body(self) -> Union[Expr, str]:
        return self.args[1]  # type: ignore

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[2:]  # type: ignore

    def toRosette(
        self,
        writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None,
        is_uninterp: bool = False,
    ) -> str:
        if self.args[1] is None and is_uninterp:  # uninterpreted function
            args_type = " ".join(["%s" % toRosetteType(a.type) for a in self.args[2:]])
            return "(define-symbolic %s (~> %s %s))" % (
                self.args[0],
                args_type,
                toRosetteType(self.returnT()),
            )

        else:
            args = " ".join(
                [
                    "%s" % (a.name)
                    if isinstance(a, ValueRef) and a.name != ""
                    else "%s" % (a.toRosette(writeChoicesTo))
                    for a in self.arguments()
                ]
            )

            return "(define-bounded (%s %s) \n%s)" % (
                self.args[0],
                args,
                self.args[1].toRosette(),
            )

    def toSMT(self) -> str:
        if self.args[1] is None:  # uninterpreted function
            args_type = " ".join(parse_type_ref_to_obj(a.type).toSMTType(get_args(a.type)) for a in self.args[2:])
            ret_type = self.returnT()
            return "(declare-fun %s (%s) %s)" % (
                self.args[0],
                args_type,
                ret_type.toSMTType(get_args(ret_type)),
            )
        else:
            declarations = []
            for a in self.args[2:]:
                if isinstance(a, ValueRef):
                    declarations.append((a.name, parse_type_ref_to_obj(a.type)))
                elif isinstance(a, NewObject):
                    declarations.append((a.src.args[0], a.type))
                else:
                    declarations.append((a.args[0], a.type))

            # TODO(jie): maybe instead of get_args(d[1]) we use containedT?
            args = " ".join(
                "(%s %s)" % (d[0], d[1].toSMTType(get_args(d[1]))) for d in declarations
            )

            if self.type.name != "Function":
                return_type = self.type.toSMT()
            else:
                return_object_type = self.returnT()
                return_type = return_object_type.toSMTType(get_args(return_object_type))
            return "(define-fun-rec %s (%s) %s\n%s)" % (
                self.args[0],
                args,
                return_type,
                self.args[1] if isinstance(self.args[1], str) else self.args[1].toSMT(),
            )

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_FnDeclRecursive(self)


class FnDefine(Expr):
    def __init__(self, name: str, returnT: Type, *args: Expr) -> None:
        Expr.__init__(self, FnT(returnT, *[a.type for a in args]), [name, *args])

    def name(self) -> str:
        return self.args[0]  # type: ignore

    def returnT(self) -> Type:
        return self.type.args[0]

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[1:]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        return ""  # only for verification

    def toSMT(self) -> str:
        args_type = " ".join(parse_type_ref_to_obj(a.type).toSMT() for a in self.args[2:])
        return "(declare-fun %s (%s) %s)" % (
            self.args[0],
            args_type,
            parse_type_ref_to_obj(self.type),
        )

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_FnDefine(self)


class Lambda(Expr):
    def __init__(self, returnT: Type, body: Expr, *args: Expr) -> None:
        Expr.__init__(self, FnT(returnT, *[a.type for a in args]), [body, *args])

    def body(self) -> Expr:
        return self.args[0]  # type: ignore

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[1:]  # type: ignore

    def toRosette(
        self, writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None
    ) -> str:
        args = " ".join(
            [
                "%s" % (a.name)
                if isinstance(a, ValueRef) and a.name != ""
                else "%s" % (a.args[0])
                for a in self.args[1:]
            ]
        )

        return "(lambda (%s) %s)" % (
            args,
            self.args[0].toRosette(),
        )

    def toSMT(self) -> str:
        # TODO(shadaj): extract during filtering assuming no captures
        raise Exception("Lambda not supported")

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Lambda(self)


class FnDecl(Expr):
    def __init__(
        self,
        name: str,
        returnT: typing.Type[NewObject],
        body: Union[Expr, str],
        *args: Expr,
    ) -> None:
        Expr.__init__(self, FnT(returnT, *[a.type for a in args]), [name, body, *args])

    def name(self) -> str:
        return self.args[0]  # type: ignore

    def returnT(self) -> typing.Type[NewObject]:
        return self.type.args[0]

    def body(self) -> Union[Expr, str]:
        return self.args[1]  # type: ignore

    def arguments(self) -> typing.List[Expr]:  # avoid name clash with Expr.args
        return self.args[2:]  # type: ignore

    def toRosette(
        self,
        writeChoicesTo: typing.Optional[Dict[str, "Expr"]] = None,
        is_uninterp: bool = False,
    ) -> str:
        if self.args[1] is None and is_uninterp:  # uninterpreted function
            args_type = " ".join(["%s" % toRosetteType(a.type) for a in self.args[2:]])
            return "(define-symbolic %s (~> %s %s))" % (
                self.args[0],
                args_type,
                toRosetteType(self.returnT()),
            )

        else:
            args = " ".join(
                [
                    "%s" % (a.name)
                    if isinstance(a, ValueRef) and a.name != ""
                    else "%s" % (a.toRosette(writeChoicesTo))
                    for a in self.arguments()
                ]
            )
            return "(define (%s %s) \n%s)" % (
                self.args[0],
                args,
                self.args[1].toRosette(),
            )

    def toSMT(self) -> str:
        if self.args[1] is None:  # uninterpreted function
            args_type = " ".join(parse_type_ref_to_obj(a.type).toSMT() for a in self.args[2:])
            return "(declare-fun %s (%s) %s)" % (
                self.args[0],
                args_type,
                parse_type_ref_to_obj(self.returnT()),
            )
        else:
            declarations = []
            for a in self.args[2:]:
                if isinstance(a, ValueRef):
                    declarations.append((a.name, parse_type_ref_to_obj(a.type)))
                else:
                    declarations.append((a.args[0], a.type))

            args = " ".join("(%s %s)" % (d[0], d[1].toSMT()) for d in declarations)

            return "(define-fun %s (%s) %s\n%s)" % (
                self.args[0],
                args,
                (
                    self.type if self.type.name != "Function" else self.type.args[0]
                ).toSMT(),
                self.args[1] if isinstance(self.args[1], str) else self.args[1].toSMT(),
            )

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_FnDecl(self)


class TargetCall(Call):
    _codegen: Optional[Callable[[Expr], str]]

    def __init__(
        self,
        name: str,
        retT: Type,
        codegen: Optional[Callable[[Expr], str]],
        *args: Expr,
    ) -> None:
        super().__init__(name, retT, *args)
        self._codegen = codegen

    def codegen(self) -> str:
        return self._codegen(*self.args[1:])  # type: ignore

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_TargetCall(self)


class Target(FnDecl):
    definedFns: Dict[str, "Target"] = {}  # stores all fns that have been defined so far

    semantics: Optional[Callable[[Expr], Expr]]
    _codegen: Optional[Callable[[Expr], str]]

    def __init__(
        self,
        name: str,
        argT: typing.List[Type],
        retT: Type,
        semantics: Callable[[Expr], Expr],
        codegen: Callable[[Expr], str],
    ) -> None:
        args: typing.List[Expr] = [Var(f"v{i}", a) for i, a in enumerate(argT)]
        super().__init__(name, retT, semantics(*args), *args)
        self.semantics = semantics
        self._codegen = codegen
        if name in Target.definedFns:
            raise Exception(f"{name} is already defined!")
        Target.definedFns[name] = self

    def call(self, *args: Expr) -> Call:
        return TargetCall(self.name(), self.returnT(), self._codegen, *args)

    def accept(self, v: "Visitor[T]") -> T:
        return v.visit_Target(self)


# class to represent the extra instructions that are inserted into the llvm code during analysis
class MLInst:
    class Kind:  # not defined as an enum as computeVC switches on the opcode str
        Assert = "assert"
        Assume = "assume"
        Call = "call"
        Eq = "eq"
        Havoc = "havoc"
        Load = "load"
        Not = "not"
        Or = "or"
        Return = "return"

    def __init__(
        self, opcode: str, *operands: Union["MLInst", Expr, ValueRef], name: str = ""
    ) -> None:
        self.opcode = opcode
        self.operands = operands
        self.name = name

    def __str__(self) -> str:
        prefix = "%s = " % self.name if self.name else ""

        if self.opcode == MLInst.Kind.Call:
            return "%scall %s %s(%s)" % (
                prefix,
                self.operands[0],
                self.operands[1],
                " ".join(
                    [
                        o.name if isinstance(o, ValueRef) else str(o)
                        for o in self.operands[2:]
                    ]
                ),
            )
        else:
            return "(%s %s)" % (
                self.opcode,
                " ".join(
                    [
                        o.name if isinstance(o, ValueRef) else str(o)
                        for o in self.operands
                    ]
                ),
            )


def MLInst_Assert(val: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Assert, val)


def MLInst_Assume(val: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Assume, val)


def MLInst_Call(name: str, retType: Type, *args: Union[MLInst, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Call, name, retType, *args)


def MLInst_Eq(
    v1: Union[MLInst, Expr, ValueRef], v2: Union[MLInst, Expr, ValueRef]
) -> MLInst:
    return MLInst(MLInst.Kind.Eq, v1, v2)


def MLInst_Havoc(*args: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Havoc, *args)


def MLInst_Load(val: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Load, val)


def MLInst_Not(val: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Not, val)


def MLInst_Or(val: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Or, val)


def MLInst_Return(val: Union[MLInst, Expr, ValueRef]) -> MLInst:
    return MLInst(MLInst.Kind.Return, val)


# def parse_type_ref(t: Union[Type, TypeRef]) -> Type:
#     # ty.name returns empty string. possibly bug
#     if isinstance(t, Type):
#         return t

#     tyStr = str(t)

#     if tyStr == "i64":
#         return Int()
#     elif tyStr == "i32" or tyStr == "i32*" or tyStr == "Int":
#         return Int()
#     elif tyStr == "i8*":
#         # TODO: this shouldn't be bool
#         return PointerT(Bool())
#     elif tyStr == "i1" or tyStr == "Bool":
#         return Bool()
#     elif tyStr.__contains__("IntObject"):  # TODO: better way to handle this
#         return Int()
#     elif (
#         tyStr.startswith("%struct.list*")
#         or tyStr == "%struct.list**"
#         or tyStr == "(MLList Int)"
#         or tyStr.__contains__("ListObject")  # TODO: better way to handle this
#     ):
#         return Type("MLList", Int())
#     elif tyStr.startswith("%struct.set"):
#         return SetT(Int())
#     elif tyStr == "(Function Bool)":
#         return Type("Function", Bool())
#     elif tyStr == "(Function Int)":
#         return Type("Function", Int())
#     elif tyStr.startswith("%struct.tup."):
#         retType = [Int() for i in range(int(tyStr[-2]) + 1)]
#         return TupleT(*retType)
#     elif tyStr.startswith("%struct.tup"):
#         # TODO: FIX return type for multiple values
#         return TupleT(Int(), Int())
#     elif tyStr == '%"class.std::__1::vector"*':
#         # we only support int vectors now
#         return ListT(Int())
#     else:
#         raise Exception("NYI %s" % t)


class Visitor(Generic[T]):
    @abstractmethod
    def visit_Var(self, o: Var) -> T:
        pass

    @abstractmethod
    def visit_NonTerm(self, o: NonTerm) -> T:
        pass

    @abstractmethod
    def visit_Lit(self, o: Lit) -> T:
        pass

    @abstractmethod
    def visit_Object(self, o: ObjectExpr) -> T:
        pass

    @abstractmethod
    def visit_Add(self, o: Add) -> T:
        pass

    @abstractmethod
    def visit_Sub(self, o: Sub) -> T:
        pass

    @abstractmethod
    def visit_Mul(self, o: Mul) -> T:
        pass

    @abstractmethod
    def visit_Eq(self, o: Eq) -> T:
        pass

    @abstractmethod
    def visit_Lt(self, o: Lt) -> T:
        pass

    @abstractmethod
    def visit_Le(self, o: Le) -> T:
        pass

    @abstractmethod
    def visit_Gt(self, o: Gt) -> T:
        pass

    @abstractmethod
    def visit_Ge(self, o: Ge) -> T:
        pass

    @abstractmethod
    def visit_And(self, o: And) -> T:
        pass

    @abstractmethod
    def visit_Or(self, o: Or) -> T:
        pass

    @abstractmethod
    def visit_Not(self, o: Not) -> T:
        pass

    @abstractmethod
    def visit_Implies(self, o: Implies) -> T:
        pass

    @abstractmethod
    def visit_Ite(self, o: Ite) -> T:
        pass

    @abstractmethod
    def visit_Let(self, o: Let) -> T:
        pass

    @abstractmethod
    def visit_Call(self, o: Call) -> T:
        pass

    @abstractmethod
    def visit_CallValue(self, o: CallValue) -> T:
        pass

    @abstractmethod
    def visit_Assert(self, o: Assert) -> T:
        pass

    @abstractmethod
    def visit_Constraint(self, o: Constraint) -> T:
        pass

    @abstractmethod
    def visit_Tuple(self, o: Tuple) -> T:
        pass

    @abstractmethod
    def visit_TupleGet(self, o: TupleGet) -> T:
        pass

    @abstractmethod
    def visit_Axiom(self, o: Axiom) -> T:
        pass

    @abstractmethod
    def visit_Synth(self, o: Synth) -> T:
        pass

    @abstractmethod
    def visit_Choose(self, o: Choose) -> T:
        pass

    @abstractmethod
    def visit_FnDeclRecursive(self, o: FnDeclRecursive) -> T:
        pass

    @abstractmethod
    def visit_FnDefine(self, o: FnDefine) -> T:
        pass

    @abstractmethod
    def visit_Lambda(self, o: Lambda) -> T:
        pass

    @abstractmethod
    def visit_FnDecl(self, o: FnDecl) -> T:
        pass

    @abstractmethod
    def visit_TargetCall(self, o: TargetCall) -> T:
        pass

    @abstractmethod
    def visit_Target(self, o: Target) -> T:
        pass


class ExtendedVisitor(Visitor[None]):
    def visit_Var(self, o: Var) -> None:
        pass

    def visit_NonTerm(self, o: NonTerm) -> None:
        pass

    def visit_Lit(self, o: Lit) -> None:
        pass

    def visit_Object(self, o: ObjectExpr) -> None:
        pass

    def generic_visit(self, o: Expr, args: Any = None) -> None:
        args = args if args else o.args
        for arg in args:
            arg.accept(self)

    def visit_Add(self, o: Add) -> None:
        self.generic_visit(o)

    def visit_Sub(self, o: Sub) -> None:
        self.generic_visit(o)

    def visit_Mul(self, o: Mul) -> None:
        self.generic_visit(o)

    def visit_Eq(self, o: Eq) -> None:
        self.generic_visit(o)

    def visit_Lt(self, o: Lt) -> None:
        self.generic_visit(o)

    def visit_Le(self, o: Le) -> None:
        self.generic_visit(o)

    def visit_Gt(self, o: Gt) -> None:
        self.generic_visit(o)

    def visit_Ge(self, o: Ge) -> None:
        self.generic_visit(o)

    def visit_And(self, o: And) -> None:
        self.generic_visit(o)

    def visit_Or(self, o: Or) -> None:
        self.generic_visit(o)

    def visit_Not(self, o: Not) -> None:
        self.generic_visit(o)

    def visit_Implies(self, o: Implies) -> None:
        self.generic_visit(o)

    def visit_Ite(self, o: Ite) -> None:
        self.generic_visit(o)

    def visit_Let(self, o: Let) -> None:
        self.generic_visit(o)

    def visit_Call(self, o: Call) -> None:
        self.generic_visit(o, args=o.args[1:])

    def visit_CallValue(self, o: CallValue) -> None:
        self.generic_visit(o)

    def visit_Assert(self, o: Assert) -> None:
        self.generic_visit(o)

    def visit_Constraint(self, o: Constraint) -> None:
        self.generic_visit(o)

    def visit_Tuple(self, o: Tuple) -> None:
        self.generic_visit(o)

    def visit_TupleGet(self, o: TupleGet) -> None:
        self.generic_visit(o)

    def visit_Axiom(self, o: Axiom) -> None:
        self.generic_visit(o)

    def visit_Synth(self, o: Synth) -> None:
        self.generic_visit(o, args=o.args[1:])

    def visit_Choose(self, o: Choose) -> None:
        self.generic_visit(o)

    def visit_FnDeclRecursive(self, o: FnDeclRecursive) -> None:
        self.generic_visit(o, args=o.args[1:])

    def visit_FnDefine(self, o: FnDefine) -> None:
        self.generic_visit(o, args=o.args[1:])

    def visit_Lambda(self, o: Lambda) -> None:
        self.generic_visit(o)

    def visit_FnDecl(self, o: FnDecl) -> None:
        self.generic_visit(o, args=o.args[1:])

    def visit_TargetCall(self, o: TargetCall) -> None:
        self.generic_visit(o)

    def visit_Target(self, o: Target) -> None:
        self.generic_visit(o)


class CountVarsVisitor(ExtendedVisitor):
    vars: Set[Var]

    def __init__(self) -> None:
        self.vars = set()

    def visit_Var(self, o: Var) -> None:
        self.vars.add(o)

    def visit_NonTerm(self, o: NonTerm) -> None:
        self.vars.add(o)
