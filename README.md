# OCR-09 googoo

## Requirements

- Python 3
- [PyTorch][pytorch]

All dependencies can be installed with PIP.

```sh
pip install tensorboardX tqdm pyyaml psutil
```

```sh
pip install -r requirements.txt
```

현재 검증된 GPU 개발환경으로는
- `Pytorch 1.0.0 (CUDA 10.1)`
- `Pytorch 1.4.0 (CUDA 10.0)`
- `Pytorch 1.7.1 (CUDA 11.0)`


## Supported Models

- [CRNN][arxiv-zhang18]
- [SATRN](https://github.com/clovaai/SATRN)


## Supported Data
- [Aida][Aida] (synthetic handwritten)
- [CROHME][CROHME] (online handwritten)
- [IM2LATEX][IM2LATEX] (pdf, synthetic handwritten)
- [Upstage][Upstage] (print, handwritten)




## Usage

### Training

```sh
python train.py
```


### Evaluation

```sh
python evaluate.py
```

[arxiv-zhang18]: https://arxiv.org/pdf/1801.03530.pdf
[CROHME]: https://www.isical.ac.in/~crohme/
[Aida]: https://www.kaggle.com/aidapearson/ocr-data
[Upstage]: https://www.upstage.ai/
[IM2LATEX]: http://lstm.seas.harvard.edu/latex/
[pytorch]: https://pytorch.org/

### Config

configs/SATRN.yaml에서 config 수정 가능.
```
network: model to use.
input_size:
  height: height of input image
  width: width of input image
SATRN:
  encoder:
    hidden_dim: size of hidden dimension of encoder
    filter_dim: size of intermediate dimension of feedforward network.
    layer_num: number of encoder layer.
    head_num: number of heads in multi-head attn.
  decoder:
    src_dim: size of input vector.
    hidden_dim: size of hidden dimension of decoder.
    filter_dim: size of intermediate dimension of feedforward network.
    layer_num: number of decoder layer.
    head_num: number of heads in multi-head attn.
Attention:
  src_dim: size of input vector.
  hidden_dim: size of hidden dimension of lstm.
  embedding_dim: size of embedding dimension of input.
  layer_num: number of latm layer
  cell_type: LSTM or GRU
checkpoint: file path of checkpoint
prefix: "./log/satrn"
version: ''

data:
  train:
    - "/opt/ml/input/data/train_dataset/gt.txt"
  test:
    - ""
  token_paths:
    - "/opt/ml/input/data/train_dataset/tokens.txt"  # 241 tokens
  dataset_proportions:  # proportion of data to take from train (not test)
    - 1.0
  random_split: if True, random split from train files
  test_proportions: 0.2 # only if random_split is True
  crop: True
  rgb: 1    # 3 for color, 1 for greyscale
  
batch_size: 34
num_workers: 8
num_epochs: 60
print_epochs: 1
dropout_rate: 0.1
teacher_forcing_ratio: 0.5
max_grad_norm: 2.0
seed: 1234
optimizer:
  optimizer: 'Adam' # Adam, Adadelta
  lr: 5e-4 # 1e-4
  weight_decay: 1e-4
  is_cycle: True
```

## experiment report
T1219_최현진 : https://github.com/bcaitech1/p4-fr-9-googoo/wiki/%5BT1219%5DSATRN-%EC%8B%A4%ED%97%98
T1243_김선욱 : https://github.com/23ksw10/Math-Expression-Recognition/blob/main/README.md
