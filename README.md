# SnakeAI

[简体中文](README_CN.md) | English | [日本語](README_JA.md)

This project contains the program scripts for the classic game "Snake" and an artificial intelligence agent that can play the game automatically. The intelligent agent is trained using deep reinforcement learning and includes two versions: an agent based on a Multi-Layer Perceptron (MLP) and an agent based on a Convolution Neural Network (CNN), with the latter having a higher average game score.

### File Structure

```bash
├───main
│   ├───logs
│   ├───trained_models_cnn
│   ├───trained_models_mlp
│   └───scripts
├───utils
│   └───scripts
```

The main code folder for the project is `main/`. It contains `logs/`, which includes terminal text and data curves of the training process (viewable using Tensorboard); `trained_models_cnn/` and `trained_models_mlp/` respectively contain the model weight files for the convolutional network and perceptron models at different stages, which can be used for running tests in `test_cnn.py` and `test_mlp.py` to observe the actual game performance of the two intelligent agents at different training stages.

The other folder `utils/` includes two utility scripts. `check_gpu_status/` is used to check if the GPU can be called by PyTorch; `compress_code.py` can remove all indentation and line breaks from the code, turning it into a tightly arranged single line of text for easier communication with GPT-4 when asking for code suggestions (GPT-4's understanding of code is far superior to humans and doesn't require indentation, line breaks, etc.).

## Running Guide

This project is based on the Python programming language and mainly uses external code libraries such as [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). The Python version used for running the program is 3.8.16. It is recommended to use [Anaconda](https://www.anaconda.com) to configure the Python environment. The following setup process has been tested on the Windows 11 system. The following commands are for the console/terminal (Console/Terminal/Shell).

### Environment Configuration

```bash
# Create a conda environment named SnakeAI with Python version 3.8.16
conda create -n SnakeAI python=3.8.16
conda activate SnakeAI

# [Optional] To use GPU for training, manually install the full version of PyTorch
conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# [Optional] Run the script to test if PyTorch can successfully call the GPU
python .\utils\check_gpu_status.py

# Install external code libraries
pip install -r requirements.txt
```

### Running Tests

The `main/` folder of the project contains the program scripts for the classic game "Snake", based on the [Pygame](https://www.pygame.org/news) code library. You can directly run the following command to play the game:

```bash
cd [parent folder of the project]/snake-ai/main
python .\snake_game.py
```

After completing the environment configuration, you can run `test_cnn.py` or `test_mlp.py` in the `main/` folder to test and observe the actual performance of the two intelligent agents at different training stages.

```bash
cd [parent folder of the project]/snake-ai/main
python test_cnn.py
python test_mlp.py
```

Model weight files are stored in the `main/trained_models_cnn/` and `main/trained_models_mlp/` folders. Both test scripts call the trained models by default. If you want to observe the AI performance at different training stages, you can modify the `MODEL_PATH` variable in the test scripts to point to the file path of other models.

### Training Models

If you need to retrain the models, you can run `train_cnn.py` or `train_mlp.py` in the `main/` folder.

```bash
cd [parent folder of the project]/snake-ai/main
python train_cnn.py
python train_mlp.py
```

### Viewing Curves

The project includes Tensorboard curve graphs of the training process. You can use Tensorboard to view detailed data. It is recommended to use the integrated Tensorboard plugin in VSCode for direct viewing, or you can use the traditional method:

```bash
cd [parent folder of the project]/snake-ai/main
tensorboard --logdir=logs/
```

Open the default Tensorboard service address `http://localhost:6006/` in your browser to view the interactive curve graphs of the training process.

## Acknowledgements
The external code libraries used in this project include [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). Thanks all the software developers for their selfless dedication to the open-source community!

The convolutional neural network used in this project is from the Nature paper:

[1] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
