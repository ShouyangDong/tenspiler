import textwrap
from typing import Any, Dict, Tuple, Union

from metalift.ir import (
    Add,
    And,
    Bool,
    Call,
    Choose,
    Div,
    Eq,
    Expr,
    FnDecl,
    FnDeclRecursive,
    Ge,
    Gt,
    Int,
    Ite,
    Le,
)
from metalift.ir import List as mlList
from metalift.ir import Lit, Lt, Matrix, Mod, Mul, Not, ObjectT, Or, Sub, Var
from tenspiler.codegen.utils import DataType
from tenspiler.tenspiler_common import (
    MAP_INT_TO_INT,
    MATRIX_ELEMWISE_ADD,
    MATRIX_ELEMWISE_DIV,
    MATRIX_ELEMWISE_MUL,
    MATRIX_ELEMWISE_SUB,
    MATRIX_SCALAR_ADD,
    MATRIX_SCALAR_DIV,
    MATRIX_SCALAR_MUL,
    MATRIX_SCALAR_SUB,
    SCALAR_MATRIX_DIV,
    SCALAR_MATRIX_SUB,
    SCALAR_VEC_DIV,
    SCALAR_VEC_SUB,
    VEC_ELEMWISE_ADD,
    VEC_ELEMWISE_DIV,
    VEC_ELEMWISE_MUL,
    VEC_ELEMWISE_SUB,
    VEC_SCALAR_ADD,
    VEC_SCALAR_DIV,
    VEC_SCALAR_MUL,
    VEC_SCALAR_SUB,
)

# Indentation is 4 spaces
INDENTATION = " " * 4

bangc_translation = {
    "matrix_vec_mul" :lambda processed_args: f"__bang_matmul({processed_args[0]}, {processed_args[1]})",
    "vec_elemwise_add": lambda processed_args: f"__bang_add({processed_args[0]}, {processed_args[1]})",
    "vec_elemwise_sub":lambda processed_args: f"__bang_sub({processed_args[0]}, {processed_args[1]})",
    "matrix_elemwise_mul": lambda processed_args: f"__bang_mul({processed_args[0]}, {processed_args[1]})",
}

tensorcore_translation = {
    "matrix_vec_mul":lambda processed_args: f"wmma::mma_sync({processed_args[0]}, {processed_args[1]})",
}

matrixcore_translation = {
    "matrix_vec_mul":lambda processed_args: f"__builtin_amdgcn_mfma_f32_16x16x4f32({processed_args[0]}, {processed_args[1]})",
}

avxvnni_translation = {
    "matrix_vec_mul": lambda processed_args: f"_mm512_dpbusd_epi32({processed_args[0]}, {processed_args[1]})",
}

def instruction_codegen(
    ps_fn_decl: Union[FnDecl, FnDeclRecursive],
    all_synthesized_fns: Dict[str, Expr],
    d_type: DataType = DataType.FLOAT,
    target=None
) -> str:
    def helper(translations, expr: Any, vars_to_replace: Dict[str, Expr] = {}) -> Tuple[str, ObjectT]:
        if not isinstance(expr, Expr):
            return str(expr), None
        if isinstance(expr, Choose):
            if len(expr.arguments()) == 1:
                return helper(expr.arguments()[0], vars_to_replace)
            else:
                raise ValueError("Choose with more than 1 argument not supported")
        if isinstance(expr, Call):
            processed_args = [
                helper(arg, vars_to_replace)[0] for arg in expr.arguments()
            ]
            fn_name = expr.name()
            if fn_name.endswith("matrix_selection_two_args"):
                for name, fn in all_synthesized_fns.items():
                    if name.endswith("select_two_args"):
                        select_two_args_fn_decl = fn
                if select_two_args_fn_decl is None:
                    raise ValueError("select_two_args not found")
                select_two_args_body = select_two_args_fn_decl.body()
                cond, if_then, if_else = (
                    select_two_args_body.c(),
                    select_two_args_body.e1(),
                    select_two_args_body.e2(),
                )
                select_args = select_two_args_fn_decl.arguments()[:2]
                matrix_args = expr.arguments()[:2]
                vars_to_replace: Dict[str, Expr] = {}
                for i in range(2):
                    vars_to_replace[select_args[i].name()] = matrix_args[i]
                return (
                    f"np.where({helper(cond, vars_to_replace)[0]}, {helper(if_then, vars_to_replace)[0]}, {helper(if_else, vars_to_replace)[0]})",
                    expr.type,
                )
            elif fn_name == MAP_INT_TO_INT or fn_name == "vec_map":
                map_fn_name = all_synthesized_fns[MAP_INT_TO_INT].body().name()
                if map_fn_name in {"integer_sqrt", "integer_exp"}:
                    return (
                        translations[map_fn_name](processed_args, fn_name == "vec_map"),
                        expr.type,
                    )
                else:
                    raise ValueError(f"Unknown map function name: {map_fn_name}")
            elif fn_name in translations.keys():
                if fn_name in {
                    VEC_ELEMWISE_DIV,
                    MATRIX_ELEMWISE_DIV,
                    SCALAR_VEC_DIV,
                    SCALAR_MATRIX_DIV,
                    VEC_SCALAR_DIV,
                    MATRIX_SCALAR_DIV,
                }:
                    return (
                        translations[fn_name](processed_args, d_type != DataType.FLOAT),
                        expr.type,
                    )
                return translations[fn_name](processed_args), expr.type
            elif fn_name in all_synthesized_fns.keys():
                return helper(all_synthesized_fns[fn_name].body())

            raise Exception(f"Unknown function name: {fn_name}")

        # Ite expression. Some condition are constants
        if isinstance(expr, Ite):
            cond = helper(expr.c())[0]

            if cond == "True":
                return helper(expr.e1(), vars_to_replace)
            elif cond == "False":
                return helper(expr.e2(), vars_to_replace)
            else:
                return (
                    f"{helper(expr.e1(), vars_to_replace)[0]} if {cond} else {helper(expr.e2(), vars_to_replace)[0]}",
                    expr.e1().type,
                )

        # Arithmetic operations
        processed_args = [helper(arg, vars_to_replace) for arg in expr.args]
        processed_args_types = [a[1] for a in processed_args]
        processed_args = [a[0] for a in processed_args]
        if any(isinstance(expr, cls) for cls in [Add, Sub, Mul, Div, Mod]):
            is_arg_type_int = all([a_type is Int for a_type in processed_args_types])
            ret_type = (
                Int
                if is_arg_type_int
                else [
                    a_type
                    for a_type in processed_args_types
                    if a_type is not Int and a_type is not None
                ][0]
            )
            if isinstance(expr, Div) and d_type == DataType.FLOAT:
                return translations["float_div"](processed_args), ret_type
            return translations[type(expr)](processed_args, is_arg_type_int), ret_type

        # Relational operations
        elif any(isinstance(expr, cls) for cls in [Gt, Ge, Eq, Lt, Le]):
            is_arg_type_int = all([a_type is Int for a_type in processed_args_types])
            ret_type = Bool if is_arg_type_int else mlList[Bool]
            return translations[type(expr)](processed_args, is_arg_type_int), ret_type
        elif any(isinstance(expr, cls) for cls in [And, Or, Not]):
            is_arg_type_prim = all(
                [a_type is Int or a_type is Bool for a_type in processed_args_types]
            )
            ret_type = Bool if is_arg_type_prim else mlList[Bool]
            return translations[type(expr)](processed_args, is_arg_type_prim), ret_type

        # Other
        elif isinstance(expr, Lit):
            return f"{expr.val()}", expr.type
        elif isinstance(expr, Var):
            if expr.name() in vars_to_replace:
                return helper(vars_to_replace[expr.name()], vars_to_replace)
            return expr.name(), expr.type

        return str(expr)

    ###############################
    # Begins actual Instruction generation
    ###############################
    fn_name = f"{ps_fn_decl.name()[:-3]}"
    arguments = [arg.name() for arg in ps_fn_decl.arguments()]
    arguments_str = ", ".join(arguments)
    print("[INFO]**************kernel_name: ", fn_name)
    print("[INFO]**************body: ", ps_fn_decl.body())
    if target == "bangc":
        translations = bangc_translation
    elif target == "tensorcore":
        translations = tensorcore_translation
    elif target == "matrixcore":
        translations = matrixcore_translation
    elif target == "dlboost":
        translations = avxvnni_translation
    else:
        raise RuntimeError("Unsupported deep learning platform.")
    instruction = {helper(translations, ps_fn_decl.body())[0]}
    print("[INFO]*************instruction: ", instruction)
    instruction = textwrap.dedent(instruction)
    return instruction
