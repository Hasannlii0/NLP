# AG News Classification with a Transformer Encoder

A from-scratch PyTorch implementation of a Transformer-encoder text classifier, trained on the [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) dataset to sort news headlines/descriptions into 4 categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

The tokenizer (Byte-Pair Encoding), the positional encoding, and the classifier itself are all implemented from scratch — no pretrained language models are used.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/AG_news_classification.ipynb)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Results

The model was trained for 5 epochs on a GPU runtime:

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc |
|:-----:|:----------:|:---------:|:----------:|:---------:|
| 1 | 0.6075 | 75.92% | 0.3937 | 85.75% |
| 2 | 0.3727 | 86.57% | 0.3399 | 87.67% |
| 3 | 0.3190 | 88.72% | 0.3141 | 88.79% |
| 4 | 0.2832 | 89.80% | 0.3009 | 89.11% |
| 5 | 0.2589 | 90.74% | 0.2916 | 89.63% |

**Final test set performance: Loss 0.2967, Accuracy 90.07%**

Example prediction:
```
Text: "The new smartphone features a powerful processor and advanced camera system."
Predicted Class: Sci/Tech
```

## Model architecture

A standard Transformer encoder used as a sequence classifier:

1. **Tokenizer** — a custom Byte-Pair Encoding (BPE) tokenizer (via Hugging Face `tokenizers`) trained from scratch on the training corpus, vocab size 16,000.
2. **Embedding layer** — maps token ids to 256-dim vectors.
3. **Positional encoding** — standard sinusoidal positional encoding added to the embeddings.
4. **Transformer encoder** — 3 stacked `TransformerEncoderLayer`s, 8 attention heads, feed-forward dim 512, dropout 0.1.
5. **Mean pooling** — averages token representations across the sequence dimension.
6. **Linear classifier head** — projects the pooled representation to 4 output classes.

| Hyperparameter | Value |
|---|---|
| Vocab size | 16,000 |
| Embedding dim | 256 |
| Attention heads | 8 |
| Feed-forward dim | 512 |
| Encoder layers | 3 |
| Dropout | 0.1 |
| Batch size | 32 |
| Optimizer | Adam (lr = 1e-4) |
| Loss | Cross-entropy |
| Epochs | 5 |

## Dataset

[AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) — 120,000 training samples and 7,600 test samples, evenly split across 4 classes (1=World, 2=Sports, 3=Business, 4=Sci/Tech). The notebook holds out the last 7,600 rows of the training set as a validation split.

## Project structure

```
.
├── AG_news_classification.ipynb   # Main notebook: data prep, tokenizer, model, training, inference
├── requirements.txt               # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Hasannlii0/NLP.git
cd NLP
```

### 2. Create an environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required — the notebook automatically falls back to CPU (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`), just slower.

### 3. Get the dataset

The notebook downloads the dataset directly from Kaggle's API, which requires Kaggle credentials:

1. Create a Kaggle account and go to **Account → Create New API Token** to download `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows), and make sure it's not world-readable:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Either let the notebook's first cell download it via `curl`, or fetch it yourself:
   ```bash
   kaggle datasets download -d amananandrai/ag-news-classification-dataset
   unzip ag-news-classification-dataset.zip
   ```

This produces `train.csv` and `test.csv`, which the notebook expects in its working directory (the notebook was originally written for Google Colab, where files live under `/content/` — update those paths if you're running locally, e.g. to `./train.csv`).

### 4. Run the notebook

```bash
jupyter notebook AG_news_classification.ipynb
```

Or open it directly in Google Colab using the badge at the top of this README (update the link with your own GitHub username/repo first).

## Notes & possible improvements

- The tokenizer is trained on `Description + Class Index` text rather than `Description` alone — worth experimenting with training on description-only text.
- No learning-rate scheduling or early stopping is used; both could improve results or training time.
- The classifier uses mean pooling over token representations; a `[CLS]`-token or attention-pooling approach could be compared against it.
- File paths are Colab-style (`/content/...`); swap these for relative paths if running locally.

## Acknowledgements

- Dataset: [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) on Kaggle, originally collected by Xiang Zhang et al. for the "Character-level Convolutional Networks for Text Classification" paper.
- Built with [PyTorch](https://pytorch.org/) and Hugging Face [tokenizers](https://github.com/huggingface/tokenizers).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
