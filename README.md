# 📊 SENTIMENT ANALYSIS PROJECT
## CODTECH Internship Task - Customer Review Classification

---

## 📋 PROJECT OVERVIEW

This project implements a **Sentiment Analysis Model** using Logistic Regression and TF-IDF vectorization to classify customer reviews as positive or negative. The project includes comprehensive text preprocessing, model training, evaluation, and visualization.

**Dataset**: Customer Reviews (100 samples - 50 positive, 50 negative)

**Objective**: Build and evaluate a sentiment classification model with detailed analysis

---

## 📁 PROJECT STRUCTURE

```
Sentiment_Analysis_Project/
│
├── Sentiment_Analysis.ipynb          # Main Jupyter notebook
├── sentiment_analysis_script.py      # Standalone Python script
├── SENTIMENT_ANALYSIS_GUIDE.md       # Comprehensive learning guide
├── SENTIMENT_README.md               # This file
├── requirements_sentiment.txt        # Dependencies
│
└── Generated Output Files:
    ├── sentiment_distribution.png
    ├── word_frequency.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    ├── sentiment_model.pkl
    └── tfidf_vectorizer.pkl
```

---

## 🚀 QUICK START

### **Option 1: Run Jupyter Notebook** (Recommended)

1. **Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

2. **Launch Jupyter:**
```bash
jupyter notebook
```

3. **Run the notebook:**
   - Open `Sentiment_Analysis.ipynb`
   - Execute all cells: `Cell > Run All`

### **Option 2: Run Python Script**

1. **Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. **Execute:**
```bash
python sentiment_analysis_script.py
```

3. **Output:**
   - Console displays all metrics
   - 5 visualization files saved
   - 2 model files saved

---

## 📦 DEPENDENCIES

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```



