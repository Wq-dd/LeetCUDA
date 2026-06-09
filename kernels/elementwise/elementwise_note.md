1. 宏
```c
// 函数式宏
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    ...
  }
// 字符串化宏
#define STRINGFY(str) #str //作用是把宏参数变成字符串字面量
// 例如
#define STRINGFY(str) #str
....
STRINGFY(elementwise_add_f32x4) // 得到"elementwise_add_f32x4"

TORCH_EXTENSION_NAME这个宏在python代码里load函数中传的那个name，编译后的扩展名就是这个例如elementwise_lib.so。
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0]) // p[i]<==>*(p+i)
// 1. 先对value取地址，然后让编译器强制解释为float4*的指针！
// 2. float4* p = reinterpret_cast<float4 *>(&(value));
// 3. 接着，在c/cpp里，p[0]==*(p+0)解引用。所以后面加[0]就等于是p[0]==*p(p+0)，这里的
// p[0]--> arr[0],arr[1],arr[2],arr[3], p[1]--> arr[4],arr[5],arr[6],arr[7]
// 4. 所以，整体就是 float4 data = reinterpret_cast<float4 *>(&(value))[0]; 解引用了。
```

2. python编译torch扩展
JIT-style的load
lib = load(name="elementwise_lib", ...)，这不是普通的import elementwise_lib，而是“编译 + 加载 + 返回模块对象”。返回之后，你想叫它什么变量名都行。可以写成`elementwise_lib = load(name="elementwise_lib", ...)` 

3. pybind11使用
```c
//elementwise_lib这个是不用加双引号的！！！
PYBIND11_MODULE(elementwise_lib, m) {
  m.def("elementwise_add_f32", &elementwise_add_f32, "help: elementwise_add_f32");
  m.def("elementwise_add_f32x4", &elementwise_add_f32x4, "help: elementwise_add_f32x4");
  m.def("elementwise_add_f16", &elementwise_add_f16, "help: elementwise_add_f16");
}
```

4. 代码内容
```c
__hadd:  是 CUDA 提供的 half 精度内置函数，用来做两个 half 的加法。
```
`elementwise_add_f16x8_kernel：` 这里面是做了展开，每个线程处理的数据更多了，分支判断少了，但是寄存器数量会增加。

优点：
- 需要的线程数量减少
- index 计算、分支判断、block/thread 调度相关开销更少
- 每个 thread 做更多连续数据，指令级并行可能更好。
- 对大数组，可能减少一些 overhead。

缺点：
- 每个 thread 用更多寄存器：reg_a_0~3, reg_b_0~3, reg_c_0~3。
- 寄存器压力可能上升，影响 occupancy。
- f16x8 每个 block 只有 1 个 warp，单 block 内部隐藏延迟能力弱一些，但 grid block 数更多。
- 对 memory-bound kernel，不一定展开越多越快。

`elementwise_add_f16x8_pack_kernel: ` 这个是一次性操作128-bit，注意`LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);`这个其实是将提供一个128-bit的容器，先强行将a和pack_a解释为float4*，这样编译器就会解释为一次加载128bit的数据了，它不知道这个数据是否正确，只管搬数据，后面`HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));`再重新解释为half就行。
