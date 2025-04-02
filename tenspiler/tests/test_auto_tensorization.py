import clang.cindex
import re

from metalift.frontend.llvm import Driver
from tenspiler.axioms_tenspiler import (
    matrix_vec_mul_axiom,
    reduce_sum_axiom,
    vec_scalar_mul_axiom,
)
from tenspiler.codegen.utils import DataType
from tenspiler.tree_parser import analyze_double_loops
from tenspiler.utils.synthesis_utils import run_synthesis_algorithm

def extract_pragma_info(tokens):
    """
    从 token 列表中查找 pragma 信息，例如：
      #pragma begin_matrix_multiply(batch=3, M=32, N=128, K=5)
    并返回解析的参数字典，以及 pragma 所在的 token 索引范围。
    """
    pragma_pattern = re.compile(r'#pragma\s+begin_matrix_multiply\s*\(batch=(\d+),\s*M=(\d+),\s*N=(\d+),\s*K=(\d+)\)')
    pragma_info = None
    start_tok = end_tok = None

    token_list = list(tokens)
    for i, token in enumerate(token_list):
        m = pragma_pattern.match(token.spelling)
        if m:
            pragma_info = {
                'batch': int(m.group(1)),
                'M': int(m.group(2)),
                'N': int(m.group(3)),
                'K': int(m.group(4))
            }
            start_tok = i
            break

    if pragma_info is None:
        return None, None, None

    # 查找对应的 #pragma end_matrix_multiply
    end_pattern = re.compile(r'#pragma\s+end_matrix_multiply')
    for j in range(start_tok + 1, len(token_list)):
        if end_pattern.match(token_list[j].spelling):
            end_tok = j
            break

    return pragma_info, start_tok, end_tok

def generate_vector_function(pragma_info, func_decl):
    """
    根据提取到的 pragma 信息和 AST 中的函数信息，
    生成基于 std::vector 的矩阵乘法函数代码，
    将原始函数名转换为新的函数名（例如 matmul），
    同时在注释中记录 pragma 参数。
    
    注意：这里转换较为简单，仅作为示例模板，
    实际应用中可能需要更精细地将指针操作转换为 vector 操作。
    """
    # 原始函数名（例如 bmm_kernel），生成新的函数名 matmul
    original_func_name = func_decl.spelling
    new_func_name = "matmul"
    
    batch = pragma_info['batch']
    M = pragma_info['M']
    N = pragma_info['N']
    K = pragma_info['K']
    
    # 下面的模板示例直接给出 vector 版本的矩阵乘法函数代码，
    # 实际转换时可以考虑根据原始函数体做更复杂的转换。
    vector_func = f"""#include <vector>
using namespace std;

// 提取的 pragma 信息: batch={batch}, M={M}, N={N}, K={K}
// 原始函数名称: {original_func_name}

vector<int> {new_func_name}(const vector<vector<int>>& weight, const vector<int>& input) {{
    vector<int> output;
    int m = weight.size();
    int n = input.size();
    for (int row = 0; row < m; row++) {{
        int curr = 0;
        for (int col = 0; col < n; col++) {{
            curr += weight[row][col] * input[col];
        }}
        output.push_back(curr);
    }}
    return output;
}}
"""
    return vector_func

def parse_file_with_ast(filename):
    # 根据需要设置 clang 库路径，例如：
    # clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang.so')
    index = clang.cindex.Index.create()
    tu = index.parse(filename, args=['-std=c++11'])

    # 通过 token 遍历查找 pragma 信息
    tokens = list(tu.get_tokens(extent=tu.cursor.extent))
    pragma_info, start_tok, end_tok = extract_pragma_info(tokens)
    if pragma_info is None:
        print("未找到 pragma 信息")
        return None

    print(f"提取到的 pragma 信息: {pragma_info}")

    # 遍历 AST 查找对应的函数声明（此处查找函数名为 bmm_kernel 的函数）
    func_decl = None
    for node in tu.cursor.get_children():
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            if node.spelling == "bmm_kernel":
                func_decl = node
                break

    if func_decl is None:
        print("未找到目标函数")
        return None

    # 输出原始函数部分信息
    print("提取到的函数信息:")
    print(f"函数名称: {func_decl.spelling}")
    print(f"返回类型: {func_decl.result_type.spelling}")
    print("参数:")
    for arg in func_decl.get_arguments():
        print(f"  {arg.spelling}: {arg.type.spelling}")

    # 此处也可利用 func_decl.extent 获取原始函数代码
    # 但转换示例中，我们使用模板生成 vector 版本代码
    vector_code = generate_vector_function(pragma_info, func_decl)
    print("转换后的 vector 形式的函数代码:")
    print(vector_code)
    return vector_code

if __name__ == "__main__":
    # 假设原始代码存储在 input.cpp 中
    vector_code = parse_file_with_ast("input.cpp")
    if vector_code is None:
        exit(1)

    # 将生成的 vector 版本的函数代码传给 analyze_double_loops 进行进一步处理
    driver, input_vars, matmul = analyze_double_loops(
        file_path=vector_code,  # 此处传入生成的 vector 函数代码
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
