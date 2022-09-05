# MLInfras: Neural Network Inference Framework

特性：
- 支持：Conv2d、ResNet BasicBlock、Linear、MaxPool2d、BatchNorm2d、GAP、Softmax等神经网络模块（参考infras/myresnet.h）
- 基于C++实现；兼容PyTorch训练框架
- 高效的张量（Tensor，参考infras/myresnet.h）计算；支持张量级优化
- 基于[OpenBLAS](https://github.com/xianyi/openblas)线性运算库的底层自主实现
- 预置ResNet32网络，并在[GTSRB](https://benchmark.ini.rub.de/)数据集上完成测试

Features: 
- Support: Conv2d, ResNet BasicBlock, Linear, MaxPool2d, BatchNorm2d, GAP, Softmax, etc. (refer to infras/myresnet.h)
- Implemented in C++; Compatible with PyTorch training framework
- Efficient Tensor computations (refer to infras/tensor.h); Tensor-level optimization supported
- Low-level customization: [OpenBLAS](https://github.com/xianyi/openblas)-based
- ResNet32 implemented, tested on [GTSRB](https://benchmark.ini.rub.de/) dataset


## 程序运行 Usage

### 准备工作 Preliminaries

```bash
make
```

或者 or

```bash
make openblas # install OpenBLAS and opencv (~10min)
make -C ./data/traffic/ # download GTSRB dataset (depends on your network)
make -C ./infras/ poc.out # compile the inference framework (<1min)
```

### 模型训练 Training

导出PyTorch训练的模型

```bash
cd model
python train.py
python export_model.py
```

### 模型预测 Inference

使用MLInfras完成模型预测

```bash
make -C ./infras/
```
