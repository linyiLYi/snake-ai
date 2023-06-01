# SnakeAI

[简体中文](README_CN.md) | [English](README.md) | 日本語

このプロジェクトは、クラシックゲーム "Snake" のプログラムスクリプトと、ゲームを自動的にプレイできる人工知能エージェントを含んでいます。人工知能エージェントは、深層強化学習を用いて学習され、以下の2つのバージョンがあります: MLP（多層パーセプトロン）を用いたエージェントとCNN（畳み込みニューラルネットワーク）を用いたエージェントがあり、後者の方が平均ゲームスコアが高いという結果になりました。

### ファイル構成

```bash
├───main
│   ├───logs
│   ├───trained_models_cnn
│   ├───trained_models_mlp
│   └───scripts
├───utils
│   └───scripts
```

プロジェクトのメインコードフォルダは `main/` です。このフォルダには `logs/` があり、学習プロセスのターミナルテキストとデータ曲線が含まれています（Tensorboard を使用して表示可能）。`trained_models_cnn/` と `trained_models_mlp/` にはそれぞれ異なるステージの畳み込みネットワークとパーセプトロンモデル用のモデル重みファイルがあり、異なる学習ステージにおける2つの知的エージェントの実際のゲーム性能を観察するために `test_cnn.py` と `test_mlp.py` でテストを行うために使用することができます。

もう一つのフォルダ `utils/` には2つのユーティリティスクリプトが含まれています。`check_gpu_status/` はGPUがPyTorchから呼び出せるかどうかをチェックするのに使われます。`compress_code.py` はコードからインデントと改行をすべて取り除き、タイトに配置した1行のテキストにして、コードの提案を求めるときに GPT-4 に伝えやすくします（GPT-4 のコードに対する理解は人間よりはるかに優れており、インデントや改行を必要としない）。

## 実行ガイド

本プロジェクトは、プログラミング言語 Python をベースに、主に [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) などの外部コードライブラリを利用しています。プログラムの実行に使用した Python のバージョンは3.8.16です。Python の環境設定には [Anaconda](https://www.anaconda.com) を使用することが推奨されます。以下の設定方法は、Windows11 のシステムで動作確認済みです。以下のコマンドは、コンソール/ターミナル（Console/Terminal/Shell）用のコマンドです。

### 環境設定

```bash
# Python バージョン3.8.16で SnakeAI という名前の conda 環境を作成
conda create -n SnakeAI python=3.8.16
conda activate SnakeAI

# [オプション] トレーニングに GPU を使用する場合は、フルバージョンの PyTorch を手動でインストール
conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# [オプション] PyTorch が GPU を正常に呼び出せるかどうかをテストするためにスクリプトを実行
python .\utils\check_gpu_status.py

# 外部コードライブラリのインストール
pip install -r requirements.txt
```

### テストの実行

プロジェクトの `main/` フォルダには、[Pygame](https://www.pygame.org/news) コードライブラリをベースにした、クラシックゲーム "Snake" のプログラムスクリプトが含まれています。以下のコマンドを直接実行することで、ゲームをプレイすることができます:

```bash
cd [parent folder of the project]/snake-ai/main
python .\snake_game.py
```

環境設定が完了したら、`main/` フォルダにある `test_cnn.py` または `test_mlp.py` を実行して、異なる学習段階における2つの知的エージェントの実際の性能をテストし観察することができます。

```bash
cd [parent folder of the project]/snake-ai/main
python test_cnn.py
python test_mlp.py
```

モデルの重みファイルは `main/trained_models_cnn/` と `main/trained_models_mlp/` フォルダに格納されます。どちらのテストスクリプトも、デフォルトで学習済みモデルを呼び出します。異なる学習段階での AI の性能を観察したい場合は、テストスクリプトの `MODEL_PATH` 変数を変更して、他のモデルのファイルパスを指すようにすることができます。

### モデルの学習

モデルの再学習が必要な場合は、`main/` フォルダにある `train_cnn.py` または `train_mlp.py` を実行することができます。

```bash
cd [parent folder of the project]/snake-ai/main
python train_cnn.py
python train_mlp.py
```

### カーブの表示

このプロジェクトには、トレーニングプロセスの Tensorboard 曲線グラフが含まれています。Tensorboard を使用すると、詳細なデータを閲覧することができます。直接見るには、VSCode に統合された Tensorboard プラグインを使用することをお勧めしますが、従来の方法を使用することもできます:

```bash
cd [parent folder of the project]/snake-ai/main
tensorboard --logdir=logs/
```

Tensorboard のデフォルトのサービスアドレス `http://localhost:6006/` をブラウザで開くと、学習過程のインタラクティブな曲線グラフが表示されます。

## 謝辞
このプロジェクトで使用した外部コードライブラリは、[Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) です。オープンソースコミュニティへの無私の献身に感謝するすべてのソフトウェア開発者に感謝します！

今回使用した畳み込みニューラルネットワークは、Nature の論文から:

[1] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
