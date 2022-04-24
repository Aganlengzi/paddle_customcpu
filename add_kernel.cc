#include "paddle/phi/extension.h" // 自定义Kernel依赖头文件

namespace custom_cpu {

// Kernel函数体实现
template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  // 使用dev_ctx的Alloc API为输出参数out分配模板参数T数据类型的内存空间
  dev_ctx.template Alloc<T>(out);
  // 使用DenseTensor的numel API获取Tensor元素数量
  auto numel = x.numel();
  // 使用DenseTensor的data API获取输入参数x的模板参数T类型的数据指针
  auto x_data = x.data<T>();
  // 使用DenseTensor的data API获取输入参数y的模板参数T类型的数据指针
  auto y_data = y.data<T>();
  // 使用DenseTensor的data API获取输出参数out的模板参数T类型的数据指针
  auto out_data = out->data<T>();
  // 完成计算逻辑
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

} // namespace custom_cpu

// 全局命名空间内使用注册宏完成Kernel注册
// CustomCPU的AddKernel注册
// 参数： add - Kernel名称
//       CustomCPU - 后端名称
//       ALL_LAYOUT - 内存布局
//       custom_cpu::AddKernel - Kernel函数名
//       int - 数据类型名
//       int64_t - 数据类型名
//       float - 数据类型名
//       double - 数据类型名
//       phi::dtype::float16 - 数据类型名
PD_REGISTER_PLUGIN_KERNEL(add,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::AddKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16){}
