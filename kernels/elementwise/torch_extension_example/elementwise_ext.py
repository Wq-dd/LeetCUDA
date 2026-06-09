import time

import torch

try:
    from elementwise_cuda import _C as lib
except ImportError as exc:
    raise SystemExit(
        "elementwise_cuda._C is not built. Run: python setup.py build_ext --inplace"
    ) from exc


torch.set_grad_enabled(False)


def benchmark(func, a, b, out, name, warmup=5, iters=20):
    for _ in range(warmup):
        func(a, b, out)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        func(a, b, out)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    print(f"{name:>24}: {elapsed_ms:.6f} ms")


def check_and_benchmark(func, a, b, name, atol, rtol):
    out = torch.empty_like(a)
    func(a, b, out)
    torch.testing.assert_close(out, torch.add(a, b), atol=atol, rtol=rtol)
    benchmark(func, a, b, out, name)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this example")

    torch.manual_seed(0)
    shape = (512, 4096)

    a_f32 = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    b_f32 = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    check_and_benchmark(lib.elementwise_add_f32, a_f32, b_f32, "f32", 1e-6, 1e-6)
    check_and_benchmark(
        lib.elementwise_add_f32x4, a_f32, b_f32, "f32x4", 1e-6, 1e-6
    )

    a_f16 = a_f32.half().contiguous()
    b_f16 = b_f32.half().contiguous()
    check_and_benchmark(lib.elementwise_add_f16, a_f16, b_f16, "f16", 1e-3, 1e-3)
    check_and_benchmark(
        lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", 1e-3, 1e-3
    )
    check_and_benchmark(
        lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", 1e-3, 1e-3
    )
    check_and_benchmark(
        lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8_pack", 1e-3, 1e-3
    )


if __name__ == "__main__":
    main()
