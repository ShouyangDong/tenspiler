import os
import subprocess

from tenspiler.codegen.numpy_codegen import numpy_codegen

def compile_all(compile_dirs):
    for d in compile_dirs:
        cwd = os.getcwd()
        os.chdir(d)
        subprocess.run(["./compile-add-blocks.sh", "ALL"], check=True)
        os.chdir(cwd)
        print("successfully compiled all input files")

compile_dirs = ["tenspiler/cpp_test"]
compile_all(compile_dirs)
