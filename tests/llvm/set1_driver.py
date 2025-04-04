from collections import defaultdict
from typing import List

from metalift.frontend.llvm import Driver, InvGrammar
from metalift.ir import Bool, Int, Object
from metalift.ir import Set as mlSet
from metalift.ir import choose, fn_decl_recursive, ite
from tests.python.utils.utils import codegen


def target_lang():
    x = Int("x")
    double = fn_decl_recursive("double", Int, (x + x).src, x.src)
    return [double]


def inv_grammar(
    writes: List[Object], reads: List[Object], in_scope: List[Object], relaxed: bool
) -> Bool:
    raise Exception("no invariant")


def ps_grammar(
    writes: List[Object], reads: List[Object], in_scope: List[Object], relaxed: bool
) -> Bool:
    ret_val = writes[0]
    input_s = reads[0]
    input_add = reads[1]
    input_value = reads[2]
    output_var = writes[0]

    empty_set = mlSet.empty(Int)

    int_lit = choose(Int(0), Int(1), Int(2), Int(3))
    int_value = choose(input_value, int_lit)

    condition = input_add == int_lit

    set_in = choose(input_s, empty_set, mlSet.singleton(int_value))

    set_transform = choose(set_in, set_in.union(set_in), set_in.difference(set_in))

    chosen_transform = ite(condition, set_transform, set_transform)

    summary = output_var == chosen_transform
    return summary


if __name__ == "__main__":
    driver = Driver()
    test = driver.analyze(
        llvm_filepath="tests/llvm/set1.ll",
        loops_filepath="tests/llvm/set1.loops",
        fn_name="test",
        target_lang_fn=target_lang,
        inv_grammars=defaultdict(lambda: InvGrammar(inv_grammar, [])),
        ps_grammar=ps_grammar,
    )

    s = mlSet(Int, "s")
    add = Int("add")
    value = Int("value")
    driver.add_var_objects([s, add, value])

    test(s, add, value)

    driver.synthesize(filename="set1")
    print("\n\ngenerated code:" + test.codegen(codegen))
