"""
SENTIMENT ANALYSIS MODEL - CODTECH INTERNSHIP TASK
===================================================
Customer Review Classification using Logistic Regression and TF-IDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Stopwords list
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
])


def create_dataset():
    """Create sample customer review dataset"""
    print("=" * 60)
    print("CREATING DATASET")
    print("=" * 60)
    
    positive_reviews = [
        "This product is amazing! I absolutely love it.",
        "Excellent quality and fast shipping. Highly recommended!",
        "Best purchase I've made in years. Very satisfied!",
        "Outstanding product! Exceeded my expectations.",
        "Great value for money. Would buy again!",
        "Fantastic! Works perfectly as described.",
        "I'm very happy with this purchase. Top quality!",
        "Superb product! Exactly what I needed.",
        "Brilliant! This is the best product ever.",
        "Amazing quality and great customer service!",
        "Love it! Perfect for my needs.",
        "Wonderful product! Highly recommend to everyone.",
        "Excellent! Worth every penny.",
        "Very pleased with the quality and performance.",
        "Great product! Fast delivery too.",
        "Perfect! Just what I was looking for.",
        "Outstanding value! Really impressed.",
        "This is incredible! Best decision ever.",
        "Awesome product! Super happy with it.",
        "Fantastic quality! Will definitely buy again.",
        "Brilliant purchase! Couldn't be happier.",
        "Amazing! Better than I expected.",
        "Excellent product! Very well made.",
        "Great buy! Highly satisfied.",
        "Love this! Perfect in every way.",
    ] * 2
    
    negative_reviews = [
        "Terrible product. Waste of money!",
        "Very disappointed. Poor quality.",
        "Don't buy this! Complete garbage.",
        "Awful! Stopped working after one day.",
        "Horrible experience. Would not recommend.",
        "Worst purchase ever. Total disappointment.",
        "Poor quality. Not worth it.",
        "Bad product. Broke immediately.",
        "Disappointing! Not as described.",
        "Terrible! Complete waste of time and money.",
        "Very poor quality. Regret buying.",
        "Awful! Nothing like the pictures.",
        "Horrible! Doesn't work at all.",
        "Worst product! Avoid at all costs.",
        "Bad quality. Fell apart quickly.",
        "Disappointing purchase. Not recommended.",
        "Terrible! Save your money.",
        "Poor! Not what I expected.",
        "Awful quality. Very unhappy.",
        "Horrible! Completely useless.",
        "Worst! Don't waste your money.",
        "Bad! Stopped working immediately.",
        "Disappointing! Poor construction.",
        "Terrible quality! Regret it.",
        "Very poor! Not worth the price.",
    ] * 2
    
    reviews = positive_reviews + negative_reviews
    sentiments = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset created: {len(df)} reviews")
    print(f"Positive: {sum(sentiments)}, Negative: {len(sentiments) - sum(sentiments)}")
    
    return df


def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    words = text.split()
    words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return ' '.join(words)


def explore_data(df):
    """Perform EDA and visualizations"""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    
    df['review_length'] = df['review'].apply(len)
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sentiment_counts = df['sentiment'].value_counts()
    axes[0].bar(['Negative', 'Positive'], sentiment_counts.values, 
                color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(sentiment_counts.values):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    axes[1].pie(sentiment_counts.values, labels=['Negative', 'Positive'],
                autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90)
    axes[1].set_title('Sentiment Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sentiment distribution saved as 'sentiment_distribution.png'")
    
    return df


def preprocess_data(df):
    """Apply text preprocessing"""
    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)
    
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    print("\nPreprocessing complete!")
    print("\nExample:")
    print(f"Original: {df['review'].iloc[0]}")
    print(f"Cleaned:  {df['cleaned_review'].iloc[0]}")
    
    return df


def analyze_words(df):
    """Analyze word frequencies"""
    print("\n" + "=" * 60)
    print("WORD FREQUENCY ANALYSIS")
    print("=" * 60)
    
    def get_top_words(text_series, n=15):
        all_words = ' '.join(text_series).split()
        word_freq = Counter(all_words)
        return word_freq.most_common(n)
    
    positive_words = get_top_words(df[df['sentiment'] == 1]['cleaned_review'])
    negative_words = get_top_words(df[df['sentiment'] == 0]['cleaned_review'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    pos_words, pos_counts = zip(*positive_words)
    axes[0].barh(pos_words, pos_counts, color='#4ECDC4')
    axes[0].set_xlabel('Frequency', fontsize=12)
    axes[0].set_title('Top Words in Positive Reviews', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    neg_words, neg_counts = zip(*negative_words)
    axes[1].barh(neg_words, neg_counts, color='#FF6B6B')
    axes[1].set_xlabel('Frequency', fontsize=12)
    axes[1].set_title('Top Words in Negative Reviews', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('word_frequency.png', dpi=300, bbox_inches='tight')
    print("\n✓ Word frequency analysis saved as 'word_frequency.png'")


def vectorize_text(X_train, X_test):
    """Apply TF-IDF vectorization"""
    print("\n" + "=" * 60)
    print("TF-IDF VECTORIZATION")
    print("=" * 60)
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Testing matrix shape: {X_test_tfidf.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer


def train_model(X_train_tfidf, y_train):
    """Train Logistic Regression model"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    
    lr_model.fit(X_train_tfidf, y_train)
    
    print("✓ Model trained successfully!")
    print(f"Number of iterations: {lr_model.n_iter_[0]}")
    
    return lr_model


def evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    y_train_pred = model.predict(X_train_tfidf)
    y_test_pred = model.predict(X_test_tfidf)
    y_test_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred,
                                target_names=['Negative', 'Positive']))
    
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#4ECDC4', linewidth=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curve saved as 'roc_curve.png'")
    
    print(f"\nAUC-ROC Score: {roc_auc:.4f}")
    
    return y_test_pred


def analyze_features(model, vectorizer):
    """Analyze feature importance"""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    top_positive = feature_importance.nlargest(15, 'coefficient')
    top_negative = feature_importance.nsmallest(15, 'coefficient')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].barh(top_positive['feature'], top_positive['coefficient'], color='#4ECDC4')
    axes[0].set_xlabel('Coefficient Value', fontsize=12)
    axes[0].set_title('Top Positive Indicators', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(top_negative['feature'], top_negative['coefficient'], color='#FF6B6B')
    axes[1].set_xlabel('Coefficient Value', fontsize=12)
    axes[1].set_title('Top Negative Indicators', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance saved as 'feature_importance.png'")


def save_model(model, vectorizer):
    """Save trained model and vectorizer"""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✓ Model saved as 'sentiment_model.pkl'")
    print("✓ Vectorizer saved as 'tfidf_vectorizer.pkl'")


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS PROJECT")
    print("CODTECH INTERNSHIP TASK")
    print("=" * 60)
    
    df = create_dataset()
    
    df = explore_data(df)
    
    df = preprocess_data(df)
    
    analyze_words(df)
    
    X = df['cleaned_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    
    model = train_model(X_train_tfidf, y_train)
    
    y_test_pred = evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    analyze_features(model, vectorizer)
    
    save_model(model, vectorizer)
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("1. sentiment_distribution.png")
    print("2. word_frequency.png")
    print("3. confusion_matrix.png")
    print("4. roc_curve.png")
    print("5. feature_importance.png")
    print("6. sentiment_model.pkl")
    print("7. tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
