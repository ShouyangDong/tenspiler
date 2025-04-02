from metalift.ir import Int, List, Matrix, fn_decl_recursive, ite
from tenspiler.codegen.instruction_codegen import instruction_codegen
from tenspiler.codegen.utils import DataType
from tenspiler.tenspiler_common import (
    DISSOLVE_MATRIX_SELECTION_TWO_ARGS,
    DISSOLVE_SELECT_TWO_ARGS,
    MAP_INT_TO_INT,
    MATRIX_SELECTION_TWO_ARGS,
    SELECT_TWO_ARGS,
    call_dissolve_matrix_selection_two_args,
    call_integer_exp,
    call_integer_sqrt,
    call_map_int_to_int,
    call_matrix_selection_two_args,
    call_matrix_vec_mul,
    call_reduce_max,
    call_reduce_sum,
    call_vec_map,
    dissolve_matrix_selection_two_args_fn_decl,
    dissolve_select_two_args_fn_obj_arg,
    map_int_to_int_fn_obj,
    matrix_selection_two_args_fn_decl,
    select_two_args_fn_obj_arg,
)

def matmul():
    weight = Matrix(Int, "weight")
    input = List(Int, "input")
    fn_decl = fn_decl_recursive(
        "matmul_ps",
        List[Int],
        call_matrix_vec_mul(weight, input),
        weight,
        input,
    )
    all_fn_decls = {"matmul_ps": fn_decl}
    return fn_decl, all_fn_decls, DataType.FLOAT

codegen_funcs = instruction_codegen
for target in ["bangc", "tensorcore", "matrixcore", "dlboost"]:
    codegen_funcs(*matmul(), target)
