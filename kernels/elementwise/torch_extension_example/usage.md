# Elementwise CUDA Extension Build

This directory includes a standalone PyTorch CUDA extension package example for
the existing `elementwise.cu` kernels.

## Build

```bash
cd kernels/elementwise
python setup.py build_ext --inplace
```

The compiled extension is placed under the Python package directory:

```text
kernels/elementwise/elementwise_cuda/_C*.so
```

You can also install it in editable mode:

```bash
cd kernels/elementwise
pip install -e .
```

`setup.py` respects `TORCH_CUDA_ARCH_LIST` when it is already set. If the
environment variable is not set, the example provides a small default arch list
for recent NVIDIA GPUs.

## Run

```bash
cd kernels/elementwise
python elementwise_ext.py
```

The example imports the built `elementwise_cuda._C` module, checks the
extension results against `torch.add`, and prints a short benchmark for the f32
and f16 elementwise add kernels.
