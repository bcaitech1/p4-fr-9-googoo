# 업스테이지 수학 수식 OCR 모델

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

## experiment report
T1219_최현진 : https://github.com/bcaitech1/p4-fr-9-googoo/wiki/%5BT1219%5DSATRN-%EC%8B%A4%ED%97%98
