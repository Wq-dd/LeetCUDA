### 内置函数
```
__hgt (half greater-than)(x1, x2):      x1  > x2
__hlt (half less-than)(x1, x2):         x1  < x2
__hge (half greater-equal)(x1, x2):     x1 >= x2
__hle (half less-equal)(x1, x2):        x1 <= x2
__heq (half equal)(x1, x2):             x1 == x2
__hne (half not equal)(x1, x2):         x1 != x2
```

### 优化trick
1. `__device__ __forceinline__ <data_type> <func_name>(args...)`: 只能从gpu上调用的函数，且强制内联。

### ncu profile结论

看来fp16_pack_kernel和torch的实现，原始代码没有用nn.functional.elu(已做算子融合)。

torch elu:
- compute利用率很高
- torch的通用模板因为要处理多种数据类型，处理复杂的if/else逻辑，会产生很多冗余指令，它虽然让 SM 满载，但其中很大一部分指令是在做“管理工作”而非核心计算。
- torch的指令过多会导致mem利用率低
- 寄存器使用量32

fp16_pack_kernel:
- 寄存器使用19个，kernel轻量，gpu的warp调度器可以很容易切换其他活跃线程。
- __hgt有分支预测，谓词执行允许 GPU 同时计算分支的两面，然后根据条件丢弃结果，这在分支简单的算子中比传统的 if/else 跳转快得多。


