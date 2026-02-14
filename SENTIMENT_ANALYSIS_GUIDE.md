# SENTIMENT ANALYSIS - LEARNING GUIDE
## CODTECH Internship Task

---

## 📚 TABLE OF CONTENTS

1. [What is Sentiment Analysis?](#what-is-sentiment-analysis)
2. [Understanding the Pipeline](#understanding-the-pipeline)
3. [Text Preprocessing Explained](#text-preprocessing-explained)
4. [TF-IDF Vectorization](#tf-idf-vectorization)
5. [Logistic Regression for Text](#logistic-regression-for-text)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Interpreting Results](#interpreting-results)
8. [Common Challenges](#common-challenges)
9. [Extension Ideas](#extension-ideas)

---

## 🎯 WHAT IS SENTIMENT ANALYSIS?

**Sentiment Analysis** (also called Opinion Mining) is the process of determining the emotional tone behind text.

### Real-World Applications:

- **E-commerce**: Analyze product reviews (Amazon, Yelp)
- **Social Media**: Track brand sentiment (Twitter, Facebook)
- **Customer Service**: Prioritize negative feedback
- **Market Research**: Understand customer opinions
- **Politics**: Analyze public opinion on policies

### Types of Sentiment:

1. **Binary**: Positive vs Negative (our project)
2. **Multi-class**: Positive, Neutral, Negative
3. **Fine-grained**: 1-star to 5-star ratings
4. **Aspect-based**: Sentiment per product feature

---

## 🔄 UNDERSTANDING THE PIPELINE

Our sentiment analysis follows this flow:

```
Raw Text
    ↓
Text Preprocessing (cleaning)
    ↓
TF-IDF Vectorization (numbers)
    ↓
Logistic Regression (training)
    ↓
Predictions (positive/negative)
```

### Step-by-Step Breakdown:

**Step 1: Raw Text**
```
"This product is AMAZING!!! I love it so much 😊"
```

**Step 2: Preprocessing**
```
"product amazing love much"
```

**Step 3: TF-IDF**
```
[0.0, 0.45, 0.0, 0.67, 0.23, ...]
```

**Step 4: Model**
```
Positive (confidence: 95%)
```

---

## 🧹 TEXT PREPROCESSING EXPLAINED

### Why Preprocess?

Raw text has noise that confuses models:
- Inconsistent capitalization
- Special characters
- Common meaningless words
- Spaces and formatting

### Preprocessing Steps:

#### 1. **Lowercasing**

**Why**: "Great" and "great" should be treated the same

```python
Before: "This Product is AMAZING!"
After:  "this product is amazing!"
```

#### 2. **Remove Numbers**

**Why**: Usually don't carry sentiment

```python
Before: "Arrived in 2 days! Great!"
After:  "Arrived in days! Great!"
```

#### 3. **Remove Punctuation**

**Why**: Simplifies text, keeps words

```python
Before: "Love it!!! Best ever!!!"
After:  "Love it Best ever"
```

#### 4. **Remove Extra Spaces**

**Why**: Normalizes spacing

```python
Before: "Great    product"
After:  "Great product"
```

#### 5. **Remove Stopwords**

**What are stopwords?**: Common words that don't add meaning
- Examples: the, a, is, in, of, to, and

**Why remove them?**: Focus on meaningful words

```python
Before: "This is the best product that I have ever used"
After:  "best product ever used"
```

### Impact Example:

```
Original (53 chars):
"This is absolutely AMAZING!!! I love it so much :)"

Cleaned (19 chars):
"absolutely amazing love much"

Reduction: 64% shorter, kept all sentiment!
```

---

## 📊 TF-IDF VECTORIZATION

### The Problem:

Machine learning models need **numbers**, not text.

```
"Great product" → ❌ Can't process
[0.5, 0.8, 0.0, ...] → ✓ Can process
```

### The Solution: TF-IDF

**TF-IDF** = Term Frequency × Inverse Document Frequency

### Breaking It Down:

#### Term Frequency (TF)

**How often does a word appear in this document?**

```
Review: "Great product! Great quality! Great price!"

TF(great) = 3/7 = 0.43
TF(product) = 1/7 = 0.14
TF(quality) = 1/7 = 0.14
```

#### Document Frequency (DF)

**In how many documents does this word appear?**

```
100 reviews total:
- "great" appears in 80 reviews
- "excellent" appears in 20 reviews

DF(great) = 80/100 = 0.80
DF(excellent) = 20/100 = 0.20
```

#### Inverse Document Frequency (IDF)

**How unique/rare is this word?**

```
IDF = log(Total Documents / Documents with Word)

IDF(great) = log(100/80) = 0.22 (common word)
IDF(excellent) = log(100/20) = 1.61 (rare word)
```

#### Final TF-IDF Score

```
TF-IDF = TF × IDF

Word "great": 0.43 × 0.22 = 0.09 (low score - too common)
Word "excellent": 0.14 × 1.61 = 0.23 (higher score - more unique)
```

### Why This Works:

✅ **Common words** (the, is, a) → Low IDF → Low TF-IDF
✅ **Unique sentiment words** (amazing, terrible) → High IDF → High TF-IDF
✅ **Frequent unique words** → Highest TF-IDF scores

### N-grams:

**Unigrams** (single words): "not", "good"
**Bigrams** (word pairs): "not good"

```python
ngram_range=(1, 2)  # Both unigrams and bigrams
```

**Why bigrams?**: Captures context

```
"not good" ≠ "good"
"very disappointed" ≠ "very" + "disappointed"
```

---

## 🎲 LOGISTIC REGRESSION FOR TEXT

### What is Logistic Regression?

A classification algorithm that predicts **probabilities**.

```
Input: TF-IDF vector
Output: Probability of being positive
```

### How It Works:

#### 1. **Linear Combination**

```
Score = w₁×feature₁ + w₂×feature₂ + ... + bias
```

Example:
```
Score = 0.8×TF-IDF(amazing) + 0.6×TF-IDF(great) - 0.9×TF-IDF(terrible)
```

#### 2. **Sigmoid Function**

Converts score to probability (0 to 1):

```
Probability = 1 / (1 + e^(-score))
```

```
Score = +5  → Probability = 0.99 (very positive)
Score = 0   → Probability = 0.50 (neutral)
Score = -5  → Probability = 0.01 (very negative)
```

#### 3. **Decision Threshold**

```
If probability > 0.5 → Positive
If probability < 0.5 → Negative
```

### Feature Weights:

Each word gets a weight (coefficient):

```
Positive weights:
"amazing": +0.85
"excellent": +0.72
"love": +0.68

Negative weights:
"terrible": -0.91
"awful": -0.85
"waste": -0.73
```

**Interpretation**: 
- Positive weight → word indicates positive sentiment
- Negative weight → word indicates negative sentiment
- Weight magnitude → strength of indication

---

## 📈 EVALUATION METRICS

### 1. **Accuracy**

**What**: Percentage of correct predictions

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```
100 reviews tested
95 correctly classified
Accuracy = 95/100 = 95%
```

**Limitation**: Can be misleading with imbalanced data

### 2. **Confusion Matrix**

**Visual breakdown of predictions:**

```
                Predicted
              Neg    Pos
Actual  Neg    45     5     (50 negative)
        Pos     3    47     (50 positive)
```

**Components**:
- **True Negative (TN)**: 45 - Correctly predicted negative
- **False Positive (FP)**: 5 - Wrongly predicted positive
- **False Negative (FN)**: 3 - Wrongly predicted negative
- **True Positive (TP)**: 47 - Correctly predicted positive

### 3. **Precision**

**What**: Of all positive predictions, how many were correct?

```
Precision = TP / (TP + FP)
         = 47 / (47 + 5)
         = 90.4%
```

**When important**: Avoiding false positives
- Spam detection (don't mark good emails as spam)
- Medical diagnosis (don't misdiagnose healthy patients)

### 4. **Recall (Sensitivity)**

**What**: Of all actual positives, how many did we catch?

```
Recall = TP / (TP + FN)
       = 47 / (47 + 3)
       = 94.0%
```

**When important**: Avoiding false negatives
- Disease screening (don't miss sick patients)
- Fraud detection (don't miss fraudulent transactions)

### 5. **F1-Score**

**What**: Harmonic mean of Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.904 × 0.940) / (0.904 + 0.940)
   = 92.2%
```

**Why**: Balances both precision and recall

### 6. **ROC-AUC**

**ROC Curve**: Plots True Positive Rate vs False Positive Rate

**AUC**: Area Under the Curve

```
AUC = 1.0   Perfect classifier
AUC = 0.9+  Excellent
AUC = 0.8+  Good
AUC = 0.7+  Fair
AUC = 0.5   Random guessing
```

---

## 🔍 INTERPRETING RESULTS

### Good Model Indicators:

✅ **High Accuracy** (>90%)
✅ **Balanced Precision & Recall**
✅ **High AUC-ROC** (>0.9)
✅ **Consistent CV scores**
✅ **Train ≈ Test accuracy** (no overfitting)

### Warning Signs:

❌ **Train >> Test accuracy** → Overfitting
❌ **Low precision** → Too many false positives
❌ **Low recall** → Missing many positives
❌ **Imbalanced confusion matrix** → Bias toward one class

### Example Analysis:

```
Results:
- Training Accuracy: 98%
- Testing Accuracy: 96%
- Precision: 95%
- Recall: 94%
- F1-Score: 94.5%
- AUC: 0.98

Interpretation:
✓ Excellent performance
✓ No overfitting (2% gap)
✓ Balanced precision/recall
✓ Strong discriminative ability
```

---

## 🚧 COMMON CHALLENGES

### 1. **Sarcasm & Irony**

**Problem**: "Oh great, it broke already!" (negative but has "great")

**Solutions**:
- Use bigrams/trigrams to capture context
- Advanced models (BERT) understand context better
- Domain-specific training

### 2. **Negation**

**Problem**: "not good" vs "good"

**Solutions**:
- Use bigrams: "not good" is one feature
- Negation handling: mark words after "not"
- Advanced models understand syntax

### 3. **Domain-Specific Language**

**Problem**: "sick" means good in slang, bad in health

**Solutions**:
- Train on domain-specific data
- Use domain lexicons
- Fine-tune on your specific use case

### 4. **Imbalanced Data**

**Problem**: 90% positive, 10% negative reviews

**Solutions**:
```python
# Use class weights
class_weight='balanced'

# Stratified sampling
stratify=y in train_test_split

# Oversample minority class
from imblearn.over_sampling import SMOTE
```

### 5. **Short Text**

**Problem**: "Great!" - not much information

**Solutions**:
- Collect more context
- Use character n-grams
- Ensemble multiple signals

---

## 🚀 EXTENSION IDEAS

### 1. **Multi-Class Classification**

Add neutral sentiment:

```python
sentiments = [0, 1, 2]  # Negative, Neutral, Positive
```

### 2. **Aspect-Based Sentiment**

Sentiment per feature:

```
"Great camera, but terrible battery"
→ camera: positive, battery: negative
```

### 3. **Advanced Models**

```python
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

# Neural Networks
from sklearn.neural_network import MLPClassifier

# BERT (state-of-the-art)
from transformers import BertForSequenceClassification
```

### 4. **Deep Learning**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### 5. **Real-Time Prediction API**

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['review']
    prediction = model.predict(text)
    return {'sentiment': prediction}
```

---

## 📝 CHECKLIST

Before submission:

✅ All code runs without errors
✅ Preprocessing steps clearly shown
✅ TF-IDF vectorization explained
✅ Model trained and evaluated
✅ Confusion matrix visualized
✅ ROC curve plotted
✅ Feature importance analyzed
✅ Model saved
✅ Results interpreted
✅ Notebook well-organized

---

## 🎓 LEARNING OUTCOMES

After this project, you understand:

1. ✅ How sentiment analysis works
2. ✅ Text preprocessing techniques
3. ✅ TF-IDF vectorization
4. ✅ Logistic regression for text
5. ✅ Evaluation metrics interpretation
6. ✅ Model deployment preparation
7. ✅ Real-world NLP challenges

---

## 💡 KEY TAKEAWAYS

1. **Text preprocessing is crucial** - garbage in, garbage out
2. **TF-IDF balances frequency and uniqueness** - smart feature extraction
3. **Logistic regression is interpretable** - see which words matter
4. **Multiple metrics tell the full story** - accuracy isn't enough
5. **Context matters** - bigrams help capture it
6. **Domain knowledge improves performance** - understand your data
7. **Start simple, then improve** - baseline before complexity

---

## 📚 ADDITIONAL RESOURCES

**Books**:
- "Natural Language Processing with Python" by Bird, Klein, Loper
- "Speech and Language Processing" by Jurafsky & Martin

**Online**:
- NLTK Documentation
- Scikit-learn Text Processing Guide
- Kaggle NLP Competitions

**Datasets**:
- IMDB Movie Reviews
- Amazon Product Reviews
- Twitter Sentiment140
- Yelp Reviews

---

**CONGRATULATIONS! 🎉**

You've built a complete sentiment analysis system!

Keep practicing and exploring! 🚀
