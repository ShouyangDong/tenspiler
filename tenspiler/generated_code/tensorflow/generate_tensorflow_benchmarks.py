import os
import runpy
import sys
from os.path import basename, dirname, join
from pathlib import Path

from tenspiler.codegen.tensorflow_codegen import tensorflow_codegen
from tenspiler.codegen.utils import DataType
from tenspiler.tests.test_codegen import *


def find_all_drivers(driver_dirs):
    driver_files = []
    for root_dir in driver_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith("_driver.py"):
                    full_path = join(dirpath, filename)
                    driver_files.append(full_path)
    return driver_files


driver_dirs = ["tenspiler/c2taco/holing/driver/"]
driver_files = find_all_drivers(driver_dirs)

stored_path = "./tenspiler/generated_code/tensorflow/"

for script_path in driver_files:
    script_dir = dirname(script_path)
    script_basename = basename(script_path)
    script_name, _ = script_basename.rsplit(".", 1)

    if script_dir not in sys.path:
        sys.path.append(script_dir)

    result = runpy.run_module(script_name, run_name="__main__")

    if script_dir in sys.path:
        sys.path.remove(script_dir)

    driver = result.get("driver")

    # NOTE: All c2taco uses full integer
    tensorflow_code = tensorflow_codegen(
        driver.get_actual_ps_fn_decl(), driver.synthesized_fns, DataType.FULL_INT
    )

    e2e_fn_name = f"{driver.get_actual_ps_fn_decl().name()[:-3]}_tf"
    kernel_fn_name = e2e_fn_name + "_kernel"

    # NOTE: The following if only needed for generating timing executables and need manual effort in connecting input.
    #     setup_code = f"""
    # ####### more import statements for benchmarking ########
    # import time
    # import cv2
    # import os
    # import numpy as np

    # ####### setup for benchmarking ########
    # gpus = tf.config.list_physical_devices('GPU')
    # if not gpus:
    #   print("No GPU is available")
    # rng = np.random.default_rng(1)

    # folder = "./tenspiler/data/data_sampled"

    # img_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    # bases = []
    # actives = []

    # for _file in img_files:
    #     img = cv2.imread(_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    #     rnd = (rng.random(img.shape, dtype = np.float32) * 255).astype(np.uint8)
    #     bases.append(img)
    #     actives.append(rnd)
    # """

    #     tensorflow_code_timing = tensorflow_code + setup_code + """
    # ####### runner. need to manually update for each file ########
    # runs = 10
    # times = []
    # for _ in range(runs):
    #     total_time = 0
    #     for i in range(len(bases)):
    #         b = bases[i].flatten().astype(np.int32)
    #         a = actives[i].flatten().astype(np.int32)
    #         b = bases[i].astype(np.int32)
    #         a = actives[i].astype(np.int32)
    #         rand_f = rng.integers(low=0, high=np.iinfo(np.int32).max + 1, size=b.shape, dtype=np.int32)
    #         s = rng.integers(low=0, high=np.iinfo(np.int32).max + 1).astype(np.int32)

    #         with tf.device('/GPU:0'):
    #             b = tf.convert_to_tensor(b, np.int32)
    #             a = tf.convert_to_tensor(a, np.int32)
    #             rand_f = tf.convert_to_tensor(rand_f, np.int32)

    #             n, = b.shape
    #             m, n = b.shape

    #             start_time = time.perf_counter()

    #             end_time = time.perf_counter()

    #         total_time += (end_time - start_time) * 1000

    #     times.append(total_time)

    # times = np.array(times)
    # """

    #     tensorflow_code_timing_kernel = tensorflow_code_timing + f"""
    # print("{kernel_fn_name}")
    # print(f"{{np.average(times)}} {{np.std(times)}}")
    # """
    tensorflow_code_e2e_timing = tensorflow_code
    #     tensorflow_code_e2e_timing = tensorflow_code_timing + f"""
    # print("{e2e_fn_name}")
    # print(f"{{np.average(times)}} {{np.std(times)}}")
    # """

    path_obj = Path(script_path)
    folder_name = path_obj.parent.name if path_obj.parent.name != "driver" else ""

    kernel_filename = join(stored_path, folder_name, f"{kernel_fn_name}.py")
    e2e_filename = join(stored_path, folder_name, f"{e2e_fn_name}.py")

    kernel_filepath = Path(kernel_filename)
    e2e_filepath = Path(e2e_filename)

    kernel_filepath.parent.mkdir(parents=True, exist_ok=True)
    e2e_filepath.parent.mkdir(parents=True, exist_ok=True)

    # with kernel_filepath.open('w') as file:
    #     file.write(tensorflow_code_timing_kernel)

    with e2e_filepath.open("w") as file:
        file.write(tensorflow_code_e2e_timing)


blend_func_names = [
    normal_blend_8,
    normal_blend_f,
    dissolve_blend_8,
    darken_blend_8,
    multiply_blend_8,
    linear_burn_8,
    color_burn_8,
    lighten_blend_8,
    screen_blend_8,
    linear_dodge_8,
    color_dodge_8,
    overlay_blend_8,
]
llama_func_names = [
    softmax_part1,
    softmax_part2,
    softmax_part3,
    softmax_part4,
    rmsnorm_part1,
    rmsnorm_part2,
    matmul,
    transformer_part1,
    transformer_part2,
    transformer_part3,
    transformer_part4,
]

for func_name in blend_func_names + llama_func_names:
    ps_fn_decl, all_fn_decls, d_type = func_name()
    tensorflow_code = tensorflow_codegen(ps_fn_decl, all_fn_decls, d_type)
    e2e_fn_name = f"{func_name.__name__}_tf"
    path_obj = (
        Path(f"{stored_path}/blend/{e2e_fn_name}.py")
        if func_name in blend_func_names
        else Path(f"{stored_path}/llama/{e2e_fn_name}.py")
    )

    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w") as file:
        file.write(tensorflow_code)
