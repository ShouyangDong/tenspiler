from tenspiler.axioms_tenspiler import vec_elemwise_add_axiom, vec_elemwise_mul_axiom
from tenspiler.codegen.utils import DataType
from tenspiler.tree_parser import analyze_single_loop
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm

if __name__ == "__main__":
    op_name = "add"
    file_path = "tenspiler/c2taco/cpp/for_synthesis/darknet/mult_add_into_cpu.cc"
    if op_name == "copy":
        driver, input_vars, copy = analyze_single_loop(
            file_path=file_path,
            func_name="mult_add_into_cpu",
            axioms=[axioms_func],
        )
        X, = input_vars["N"]
        driver.add_precondition(X.len() >= N)
        copy(X)
        run_synthesis_algorithm(
            driver=driver,
            data_type=DataType.INT32,
            benchmark_name="mult_add_into_cpu",
            has_relaxed=False,
        ) 
    # element-wise operation
    elif op_name in ["add", "mul", "Sign"]:
        if op_name == "add":
            axioms_func = vec_elemwise_mul_axiom
        elif op_name == "mul":
            axioms_func = vec_elemwise_mul_axiom
            
        driver, input_vars, mult_add_into_cpu = analyze_single_loop(
            file_path=file_path,
            func_name="mult_add_into_cpu",
            axioms=[axioms_func],
        )
        N, X, Y, Z = input_vars["N"], input_vars["X"], input_vars["Y"], input_vars["Z"]
        driver.add_precondition(N >= 1)
        driver.add_precondition(X.len() >= N)
        driver.add_precondition(Y.len() >= N)
        driver.add_precondition(Z.len() >= N)
        mult_add_into_cpu(N, X, Y, Z)
        run_synthesis_algorithm(
            driver=driver,
            data_type=DataType.INT32,
            benchmark_name="mult_add_into_cpu",
            has_relaxed=False,
        )
    # pool operation
    elif op in ["sumpool", "minpool", "avgpool", "maxpool"]:
        driver, input_vars, pool = analyze_double_loops(
            file_path="tenspiler/llama/cpp/for_synthesis/matmul.cc",
            func_name="matmul",
            axioms=[
                vec_scalar_mul_axiom,
                reduce_sum_axiom,
                matrix_vec_mul_axiom,
            ],
        )
        input = input_vars["input"]
        driver.add_precondition(weight.len() > 0)
        driver.add_precondition(weight[0].len() > 0)
        driver.add_precondition(weight[0].len() == input.len())
        pool(input)
        run_synthesis_algorithm(
            driver=driver,
            data_type=DataType.UINT_8,
            benchmark_name="pool",
            has_relaxed=False,
        )
    
    # matmul operation
    elif op == "matmul":
        driver, input_vars, matmul = analyze_double_loops(
            file_path="tenspiler/llama/cpp/for_synthesis/matmul.cc",
            func_name="matmul",
            axioms=[
                vec_scalar_mul_axiom,
                reduce_sum_axiom,
                matrix_vec_mul_axiom,
            ],
        )
        weight, input = input_vars["weight"], input_vars["input"]
        driver.add_precondition(weight.len() > 0)
        driver.add_precondition(weight[0].len() > 0)
        driver.add_precondition(weight[0].len() == input.len())
        matmul(weight, input)
        run_synthesis_algorithm(
            driver=driver,
            data_type=DataType.UINT_8,
            benchmark_name="matmul",
            has_relaxed=False,
        )
