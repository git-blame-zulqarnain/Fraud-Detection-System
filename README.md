# Fraud Detection System

This project implements a fraud detection model on the Credit Card Fraud Detection dataset. The goal is to accurately identify fraudulent transactions despite heavy class imbalance using SMOTE and a Random Forest classifier.

---

## Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Size: ~144 MB (not included due to size constraints)
> Due to size constraints, `creditcard.csv` is not included in this repository.  
> Download it manually from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
> and place it in the `dataset/` folder before running the script.

- Features: 30 anonymized numeric features + `Class` target (0 = normal, 1 = fraud)


---

## ⚙ How to Use

1. Download `creditcard.csv` from Kaggle and place it in `dataset/`.
2. Install dependencies:
```bash
pip install -r requirements.txt
python fraud_detection.py
