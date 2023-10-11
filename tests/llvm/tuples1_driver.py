from typing import List

from metalift.frontend.llvm import Driver
from metalift.ir import (Add, Call, Choose, Eq, Expr, FnDecl, Int,
                         FnDeclRecursive, IntLit, IntObject, Mul, Sub, Tuple, TupleGet, TupleObject, TupleT, Var)
from tests.python.utils.utils import codegen

def tuple_mult(t):
    return Call("tuple_mult", IntObject, t)

def target_lang():
    x = TupleObject[IntObject](IntObject, "x")
    tuple_mult = FnDeclRecursive(
        "tuple_mult",
        IntObject,
        Mul(x[0], x[1]), # TODO(jie): maybe we can even rewrite this mul using *
        x
    )
    return [tuple_mult]

def inv_grammar(v: Var, writes: List[Var], reads: List[Var]) -> Expr:
    raise Exception("no invariant")

def ps_grammar(ret_val: Var, writes: List[Var], reads: List[Var]) -> Expr:
    (x, y) = reads
    summary = Choose(
        Eq(ret_val, Add(tuple_mult(Tuple(x, x)), tuple_mult(Tuple(y, y)))),
        Eq(ret_val, Sub(tuple_mult(Tuple(x, x)), tuple_mult(Tuple(y, y)))),
    )
    return summary

if __name__ == "__main__":
    driver = Driver()
    test = driver.analyze(
        llvm_filepath="tests/llvm/tuples1.ll",
        loops_filepath="tests/llvm/tuples1.loops",
        fn_name="test",
        target_lang_fn=target_lang,
        inv_grammar=inv_grammar,
        ps_grammar=ps_grammar
    )

    x = IntObject("x")
    y = IntObject("y")
    driver.add_var_objects([x, y])

    test(x, y)

    driver.synthesize()

    print("\n\ngenerated code:" + test.codegen(codegen))