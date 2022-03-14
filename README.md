# paddle_customcpu
A customcpu demo for paddlepaddle-plugin device
refer: [飞桨官网:文档:硬件支持:自定义硬件接入](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/index_cn.html)

## 前提
安装[PaddlePaddle develop最新版本](https://github.com/PaddlePaddle/Paddle)
```
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

## 编译
```bash
mkdir build
cd build
cmake ..
make
```
## 安装
```bash
pip install dist/paddle_custom_cpu-0.0.1-cp37-cp37m-linux_x86_64.whl
```

## 使用
```python
(base) python
Python 3.7.9 (default, Aug 31 2020, 12:42:55)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import paddle
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
I0314 23:55:31.851675 55839 init.cc:259] ENV [CUSTOM_DEVICE_ROOT]=/opt/conda/lib/python3.7/site-packages/paddle-plugins
I0314 23:55:31.851728 55839 init.cc:147] Try loading custom device libs from: [/opt/conda/lib/python3.7/site-packages/paddle-plugins]
I0314 23:55:31.852723 55839 custom_device.cc:710] Successed in loading custom runtime in lib: /opt/conda/lib/python3.7/site-packages/paddle-plugins/libpaddle_custom_cpu.so
I0314 23:55:31.852778 55839 custom_kernel.cc:68] Successed in loading 1 custom kernel(s) from loaded lib(s), will be used like native ones.
I0314 23:55:31.852797 55839 init.cc:159] Finished in LoadCustomDevice with libs_path: [/opt/conda/lib/python3.7/site-packages/paddle-plugins]
I0314 23:55:31.852856 55839 init.cc:265] CustomDevice: CustomCPU, visible devices count: 1
>>> paddle.device.get_all_custom_device_type()
['CustomCPU']
>>> paddle.set_device('CustomCPU')
Place(CustomCPU:0)
>>> x = paddle.to_tensor([1])
>>> x + x
Tensor(shape=[1], dtype=int64, place=Place(CustomCPU:0), stop_gradient=True,
       [2])
>>> exit()
```
