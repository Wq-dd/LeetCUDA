import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


PACKAGE_NAME = "toy-elementwise"
EXTENSION_NAME = "elementwise_cuda._C"
THIS_DIR = Path(__file__).resolve().parent


def get_cuda_version():
    if CUDA_HOME is None:
        return None
    output = subprocess.check_output(
        [str(Path(CUDA_HOME) / "bin" / "nvcc"), "-V"],
        universal_newlines=True,
    )
    version = output.split("release", 1)[1].split(",", 1)[0].strip()
    return tuple(int(part) for part in version.split(".")[:2])


def set_default_arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    arch_list = ["8.0", "8.9"]
    cuda_version = get_cuda_version()
    if cuda_version is not None and cuda_version >= (11, 8):
        arch_list.append("9.0")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


set_default_arch_list()

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="Elementwise CUDA extension example",
    packages=["elementwise_cuda"],
    ext_modules=[
        CUDAExtension(
            name=EXTENSION_NAME,
            sources=[str(THIS_DIR / "elementwise.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch", "ninja"],
)
