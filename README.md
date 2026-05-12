# LitTx: A New Treatment Relation Extraction Dataset

## Setup

### 1. Dataset Access

Our dataset is encrypted for controlled distribution.  
To decrypt it, run the following command:

```bash
cd dataset
python decrypt.py
```

---

### 2. Install Dependencies

Before running the code, install all required packages using:

```bash
pip install -r requirements.txt
```

---

### 3. Train and Evaluate Models

Update your `train_args.py` file with your [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens) to enable model downloads and authentication.  
Once updated, start training and evaluation on the **LitTx** dataset by running:

```bash
python train_args.py
```

This will automatically train the model and evaluate its performance on the dataset.

---

### 4. Annotation Guideline

Please find the information in the [annotation guideline](./LitTx_guidelines.docx).

---

## Citation

If you use **LitTx** in your research, please cite:

```bibtex
@inproceedings{jiang-etal-2026-littx,
  title = {LitTx: A New Treatment Relation Extraction Dataset},
  author = {Jiang, Yuhang and Nahian, Md Sultan Al and Xu, Li Hao Richie and Chikkanna, Rani and Kavuluru, Ramakanth},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {1352--1360}
}
```



