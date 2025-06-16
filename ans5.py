import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import random

random.seed(42)
np.random.seed(42)

def create_synthetic_dataset(n_samples=100):
    """Create a synthetic dataset of 100 short text samples with binary labels"""
    
    positive_samples = [
        "This product is amazing and works perfectly",
        "Excellent quality and fast delivery",
        "I love this item, highly recommend it",
        "Great value for money, very satisfied",
        "Outstanding service and product quality",
        "Perfect condition, exactly as described",
        "Fantastic product, exceeded my expectations",
        "Very happy with this purchase",
        "Excellent customer service and quality",
        "Amazing product, will buy again",
        "Best purchase I've made in a while",
        "Superb quality and great price",
        "Wonderful product, very pleased",
        "Excellent build quality and design",
        "Perfect fit and great material",
        "Outstanding performance and value",
        "Very impressed with the quality",
        "Great product, fast shipping",
        "Excellent condition and packaging",
        "Highly satisfied with this purchase",
        "Amazing quality for the price",
        "Perfect product, works as expected",
        "Great customer service experience",
        "Excellent value and quality",
        "Very good product, recommend it",
        "Fantastic quality and design",
        "Perfect size and great material",
        "Excellent product, very happy",
        "Great quality and fast delivery",
        "Amazing product, love it",
        "Perfect condition, great seller",
        "Excellent service and product",
        "Very satisfied with quality",
        "Great product, works perfectly",
        "Amazing value for money",
        "Perfect product, highly recommend",
        "Excellent quality and service",
        "Very impressed with this item",
        "Great purchase, very happy",
        "Amazing product quality",
        "Perfect condition and packaging",
        "Excellent build and design",
        "Very good quality product",
        "Great value and performance",
        "Amazing customer service",
        "Perfect product functionality",
        "Excellent material quality",
        "Very satisfied customer",
        "Great product experience",
        "Amazing shipping speed"
    ]
    
    negative_samples = [
        "Terrible product, doesn't work at all",
        "Very poor quality, waste of money",
        "Disappointing purchase, not as described",
        "Bad quality and slow delivery",
        "Awful product, completely useless",
        "Poor customer service and quality",
        "Terrible experience, do not buy",
        "Very bad product, broke immediately",
        "Disappointing quality for the price",
        "Awful delivery and packaging",
        "Poor build quality and design",
        "Terrible value for money",
        "Very disappointing purchase",
        "Bad quality control issues",
        "Awful product performance",
        "Poor material quality",
        "Terrible customer service",
        "Very bad shipping experience",
        "Disappointing product quality",
        "Awful packaging and condition",
        "Poor product functionality",
        "Terrible build quality",
        "Very poor value",
        "Bad customer support",
        "Awful product design",
        "Poor quality materials",
        "Terrible user experience",
        "Very disappointing quality",
        "Bad product performance",
        "Awful delivery service",
        "Poor packaging quality",
        "Terrible product condition",
        "Very bad quality control",
        "Disappointing build quality",
        "Awful customer experience",
        "Poor product reliability",
        "Terrible shipping quality",
        "Very poor construction",
        "Bad product durability",
        "Awful quality issues",
        "Poor value proposition",
        "Terrible product defects",
        "Very disappointing service",
        "Bad manufacturing quality",
        "Awful product problems",
        "Poor overall experience",
        "Terrible quality concerns",
        "Very bad product issues",
        "Disappointing customer service",
        "Awful product failures"
    ]
    
    texts = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def text_preprocess_vectorize(texts, vectorizer):
    """
    Takes a list of text samples and a fitted TfidfVectorizer,
    and returns the vectorized feature matrix.
    """
    feature_matrix = vectorizer.transform(texts)
    return feature_matrix

def main():
    print("=== Problem 5: Complete NLP Pipeline ===\n")
    
    print("Task 1: Creating synthetic dataset...")
    df = create_synthetic_dataset(100)
    print(f"Dataset created with {len(df)} samples")
    print(f"Positive samples: {sum(df['label'] == 1)}")
    print(f"Negative samples: {sum(df['label'] == 0)}")
    print("\nSample data:")
    print(df.head())
    print()
    
    print("Task 2: Preprocessing text with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=300,
        lowercase=True,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print()
    
    print("Task 3: Splitting data and training Logistic Regression...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Performance on Test Set:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print()
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print()
    
    print("Task 4: Demonstrating text_preprocess_vectorize function...")
    
    test_texts = [
        "This is an excellent product with great quality",
        "Terrible quality, very disappointed with purchase",
        "Amazing value for money, highly recommend"
    ]
    
    vectorized_features = text_preprocess_vectorize(test_texts, vectorizer)
    
    print(f"Input texts: {len(test_texts)}")
    print(f"Vectorized feature matrix shape: {vectorized_features.shape}")
    
    predictions = model.predict(vectorized_features)
    probabilities = model.predict_proba(vectorized_features)
    
    print("\nPredictions on new texts:")
    for i, text in enumerate(test_texts):
        sentiment = "Positive" if predictions[i] == 1 else "Negative"
        confidence = max(probabilities[i])
        print(f"Text: '{text}'")
        print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
        print()
    
    print("=== All tasks completed successfully! ===")

if _name_ == "_main_":
    main()