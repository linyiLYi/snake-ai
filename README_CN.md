# SnakeAI

简体中文 | [English](README.md)

本项目包含经典游戏《贪吃蛇》的程序脚本以及可以自动进行游戏的人工智能代理。该智能代理基于深度强化学习进行训练，包括两个版本：基于多层感知机（Multi-Layer Perceptron）的代理和基于卷积神经网络（Convolution Neural Network）的代理，其中后者的平均游戏分数更高。

### 文件结构

```bash
├───main
│   ├───logs
│   ├───trained_models_cnn
│   ├───trained_models_mlp
│   └───scripts
├───utils
│   └───scripts
```

项目的主要代码文件夹为 `main/`。其中，`logs/` 包含训练过程的终端文本和数据曲线（使用 Tensorboard 查看）；`trained_models_cnn/` 与 `trained_models_mlp/` 分别包含卷积网络与感知机两种模型在不同阶段的模型权重文件，用于在 `test_cnn.py` 与 `test_mlp.py` 中运行测试，观看两种智能代理在不同训练阶段的实际游戏效果。

另一个文件夹 `utils/` 包括两个工具脚本。`check_gpu_status/` 用于检查 GPU 是否可以被 PyTorch 调用；`compress_code.py` 可以将代码缩进、换行全部删去变成一行紧密排列的文本，方便与 GPT-4 进行交流，向 AI 询问代码建议（GPT-4 对代码的理解能力远高于人类，不需要缩进、换行等）。

## 运行指南

本项目基于 Python 编程语言，用到的外部代码库主要包括 [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等。程序运行使用的 Python 版本为 3.8.16，建议使用 [Anaconda](https://www.anaconda.com) 配置 Python 环境。以下配置过程已在 Windows 11 系统上测试通过。以下为控制台/终端（Console/Terminal/Shell）指令。

### 环境配置

```bash
# 创建 conda 环境，将其命名为 SnakeAI，Python 版本 3.8.16
conda create -n SnakeAI python=3.8.16
conda activate SnakeAI

# [可选] 使用 GPU 训练需要手动安装完整版 PyTorch
conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# [可选] 运行程序脚本测试 PyTorch 是否能成功调用 GPU
python .\utils\check_gpu_status.py

# 安装外部代码库
pip install -r requirements.txt
```

### 运行测试

项目 `main/` 文件夹下包含经典游戏《贪吃蛇》的程序脚本，基于 [Pygame](https://www.pygame.org/news) 代码库，可以直接运行以下指令进行游戏：

```bash
cd [项目上级文件夹]/snake-ai/main
python .\snake_game.py
```

环境配置完成后，可以在 `main/` 文件夹下运行 `test_cnn.py` 或 `test_mlp.py` 进行测试，观察两种智能代理在不同训练阶段的实际表现。

```bash
cd [项目上级文件夹]/snake-ai/main
python test_cnn.py
python test_mlp.py
```

模型权重文件存储在 `main/trained_models_cnn/` 与 `main/trained_models_mlp/` 文件夹下。两份测试脚本均默认调用训练完成后的模型。如果需要观察不同训练阶段的 AI 表现，可将测试脚本中的 `MODEL_PATH` 变量修改为其它模型的文件路径。

### 训练模型

如果需要重新训练模型，可以在 `main/` 文件夹下运行 `train_cnn.py` 或 `train_mlp.py`。

```bash
cd [项目上级文件夹]/snake-ai/main
python train_cnn.py
python train_mlp.py
```

### 查看曲线

项目中包含了训练过程的 Tensorboard 曲线图，可以使用 Tensorboard 查看其中的详细数据。推荐使用 VSCode 集成的 Tensorboard 插件直接查看，也可以使用传统方法：

```bash
cd [项目上级文件夹]/snake-ai/main
tensorboard --logdir=logs/
```

在浏览器中打开 Tensorboard 服务默认地址 `http://localhost:6006/`，即可查看训练过程的交互式曲线图。

## 鸣谢
本项目调用的外部代码库包括 [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等。感谢各位软件工作者对开源社区的无私奉献！

本项目使用的卷积神经网络来自 Nature 论文：

[1] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
