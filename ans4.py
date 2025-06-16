import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random

random.seed(42)
np.random.seed(42)

positive_reviews = [
    "This movie was absolutely fantastic! Great acting and amazing storyline.",
    "Brilliant cinematography and outstanding performances by the entire cast.",
    "A masterpiece that will be remembered for years to come.",
    "Incredible action sequences with deep emotional moments throughout.",
    "The director created something truly special with this film.",
    "Outstanding visual effects and a compelling narrative that kept me engaged.",
    "Excellent character development and superb dialogue writing.",
    "This film exceeded all my expectations with its innovative approach.",
    "A beautiful story told with passion and remarkable attention to detail.",
    "The acting was phenomenal and the plot was perfectly executed.",
    "An emotional rollercoaster with stunning visuals and great music.",
    "This movie delivers on every level - drama, action, and heart.",
    "Captivating from start to finish with memorable characters.",
    "The screenplay is brilliant and the cinematography is breathtaking.",
    "A perfect blend of entertainment and artistic expression.",
    "The performances were outstanding and the story was deeply moving.",
    "This film showcases incredible talent both in front and behind the camera.",
    "An unforgettable cinematic experience with powerful themes.",
    "The direction is flawless and every scene serves the story perfectly.",
    "A triumph in filmmaking with exceptional production values.",
    "The chemistry between the actors was electric and believable.",
    "This movie combines humor and drama in the most perfect way.",
    "Stunning visuals paired with an emotionally resonant storyline.",
    "The acting performances were nuanced and completely convincing.",
    "A well-crafted film that respects its audience's intelligence.",
    "The soundtrack perfectly complements the beautiful imagery.",
    "This is cinema at its finest with incredible attention to detail.",
    "The story unfolds beautifully with surprising twists and turns.",
    "Outstanding directing that brings out the best in every actor.",
    "A movie that successfully balances spectacle with genuine emotion.",
    "The writing is sharp and the performances are absolutely stellar.",
    "This film creates a world that feels authentic and immersive.",
    "Exceptional storytelling with characters you truly care about.",
    "The cinematography is gorgeous and the pacing is perfect.",
    "A powerful and moving film that stays with you long after.",
    "The production design is meticulous and visually stunning.",
    "This movie demonstrates the power of great collaborative filmmaking.",
    "An inspiring story brought to life by talented performers.",
    "The editing is seamless and enhances the emotional impact.",
    "A remarkable achievement in contemporary cinema.",
    "The film's themes are explored with depth and sensitivity.",
    "Outstanding special effects that serve the story wonderfully.",
    "This movie succeeds in creating genuine emotional connections.",
    "The dialogue feels natural and the situations are believable.",
    "A film that manages to be both entertaining and thought-provoking.",
    "The performances are layered and the direction is confident.",
    "This movie creates moments of pure cinematic magic.",
    "An excellent adaptation that honors its source material.",
    "The film's visual style perfectly matches its narrative tone.",
    "A satisfying and well-executed piece of entertainment."
]

negative_reviews = [
    "This movie was terrible with poor acting and a confusing plot.",
    "Boring and predictable storyline that wasted my time completely.",
    "The worst film I've seen this year with awful dialogue.",
    "Poorly directed with unconvincing performances throughout.",
    "A complete disaster that fails on every possible level.",
    "The plot makes no sense and the characters are poorly developed.",
    "Terrible special effects and a script that feels amateurish.",
    "This movie is a mess with no clear direction or purpose.",
    "Disappointing performances and a storyline full of plot holes.",
    "The acting was wooden and the dialogue was cringe-worthy.",
    "A waste of talent with a script that seems hastily written.",
    "This film lacks any emotional depth or compelling characters.",
    "The pacing is painfully slow and nothing interesting happens.",
    "Poor cinematography and editing that feels disjointed throughout.",
    "This movie fails to deliver on its promising premise.",
    "The performances feel forced and the story is unbelievable.",
    "A tedious film that drags on without any payoff.",
    "The direction is unfocused and the acting is unconvincing.",
    "This movie tries too hard but achieves very little.",
    "Weak character development and a plot that goes nowhere.",
    "The dialogue is stilted and the situations feel artificial.",
    "A disappointing effort that wastes a potentially good concept.",
    "This film is poorly executed with amateur-level production values.",
    "The story is confusing and the characters are unlikeable.",
    "Terrible pacing that makes the movie feel much longer than it is.",
    "The acting ranges from mediocre to downright bad throughout.",
    "This movie lacks coherence and fails to engage the audience.",
    "Poor writing combined with uninspired direction creates a mess.",
    "The film's attempts at humor fall completely flat.",
    "A boring and forgettable movie with no redeeming qualities.",
    "The plot is riddled with inconsistencies and logical gaps.",
    "This movie feels like it was made without any real vision.",
    "The performances are lifeless and the story is predictable.",
    "Poor production quality makes this film hard to watch.",
    "The script seems to have been written in a weekend.",
    "This movie fails to create any emotional connection with viewers.",
    "The direction is amateurish and the acting is unconvincing.",
    "A film that promises much but delivers absolutely nothing.",
    "The dialogue is painful to listen to and feels unnatural.",
    "This movie is a complete waste of everyone's time and money.",
    "The story structure is messy and hard to follow.",
    "Poor casting choices result in unconvincing character portrayals.",
    "This film lacks any sense of purpose or artistic vision.",
    "The editing is choppy and disrupts any potential flow.",
    "A movie that fails to understand its own genre conventions.",
    "The performances are flat and the story is uninspiring.",
    "This film tries to be clever but comes across as pretentious.",
    "Poor sound design and cinematography make viewing unpleasant.",
    "The movie's attempts at depth feel shallow and forced.",
    "A disappointing film that fails to live up to any expectations."
]

reviews_data = []

for review in positive_reviews:
    reviews_data.append({"Review": review, "Sentiment": "positive"})

for review in negative_reviews:
    reviews_data.append({"Review": review, "Sentiment": "negative"})

df = pd.DataFrame(reviews_data)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Movie Reviews Dataset")
print("=" * 50)
print(f"Total reviews: {len(df)}")
print(f"Positive reviews: {len(df[df['Sentiment'] == 'positive'])}")
print(f"Negative reviews: {len(df[df['Sentiment'] == 'negative'])}")
print("\nFirst 10 rows:")
print(df.head(10))

print("\n" + "=" * 70)
print("TOKENIZATION ANALYSIS")
print("=" * 70)

vectorizer = CountVectorizer(
    max_features=500,
    stop_words='english',
    lowercase=True
)

X = vectorizer.fit_transform(df['Review'])

feature_names = vectorizer.get_feature_names_out()

print(f"\nTokenization Results:")
print(f"Number of documents: {X.shape[0]}")
print(f"Number of features (tokens): {X.shape[1]}")
print(f"Total vocabulary size: {len(feature_names)}")

print(f"\nSparse matrix density: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
print(f"Average tokens per document: {X.nnz / X.shape[0]:.2f}")

token_frequencies = np.array(X.sum(axis=0)).flatten()
top_tokens_idx = np.argsort(token_frequencies)[::-1][:20]

print(f"\nTop 20 most frequent tokens:")
for i, idx in enumerate(top_tokens_idx, 1):
    token = feature_names[idx]
    frequency = token_frequencies[idx]
    print(f"{i:2d}. {token:<12} (frequency: {frequency})")

token_freq_df = pd.DataFrame({
    'Token': feature_names,
    'Frequency': token_frequencies
}).sort_values('Frequency', ascending=False)

print(f"\nToken frequency distribution:")
print(f"Mean frequency: {token_freq_df['Frequency'].mean():.2f}")
print(f"Median frequency: {token_freq_df['Frequency'].median():.2f}")
print(f"Std deviation: {token_freq_df['Frequency'].std():.2f}")

print(f"\nSample document-term matrix (first 5 documents, first 10 features):")
sample_matrix = X[:5, :10].toarray()
sample_features = feature_names[:10]

sample_df = pd.DataFrame(
    sample_matrix,
    columns=sample_features,
    index=[f"Doc_{i}" for i in range(5)]
)
print(sample_df)

print(f"\nCorresponding reviews for sample documents:")
for i in range(5):
    print(f"Doc_{i}: {df.iloc[i]['Review'][:60]}...")
    print(f"        Sentiment: {df.iloc[i]['Sentiment']}")

print(f"\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"✓ Created dataset with {len(df)} movie reviews")
print(f"✓ Balanced dataset: {len(df[df['Sentiment'] == 'positive'])} positive, {len(df[df['Sentiment'] == 'negative'])} negative")
print(f"✓ Tokenized using CountVectorizer with max_features=500")
print(f"✓ Removed English stop words")
print(f"✓ Resulting feature matrix: {X.shape[0]} documents × {X.shape[1]} features")
print(f"✓ Average document length: {X.nnz / X.shape[0]:.1f} tokens")

print(f"\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,  
    df['Sentiment'],  
    test_size=0.2,
    random_state=42,
    stratify=df['Sentiment']  
)

df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['Sentiment']
)

print(f"Training set:")
print(f"  - Total samples: {X_train.shape[0]}")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Positive reviews: {len(y_train[y_train == 'positive'])}")
print(f"  - Negative reviews: {len(y_train[y_train == 'negative'])}")

print(f"\nTesting set:")
print(f"  - Total samples: {X_test.shape[0]}")
print(f"  - Features: {X_test.shape[1]}")
print(f"  - Positive reviews: {len(y_test[y_test == 'positive'])}")
print(f"  - Negative reviews: {len(y_test[y_test == 'negative'])}")

print(f"\nSplit proportions:")
print(f"  - Training: {X_train.shape[0]/len(df)*100:.1f}%")
print(f"  - Testing: {X_test.shape[0]/len(df)*100:.1f}%")

print(f"\nClass distribution verification:")
print(f"Original dataset:")
print(df['Sentiment'].value_counts(normalize=True))
print(f"\nTraining set:")
print(y_train.value_counts(normalize=True))
print(f"\nTesting set:")
print(y_test.value_counts(normalize=True))

print(f"\nSample from training set:")
print(df_train.head(3))

print(f"\nSample from testing set:")
print(df_test.head(3))

print(f"\n" + "=" * 70)
print("MULTINOMIAL NAIVE BAYES CLASSIFIER")
print("=" * 70)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Multinomial Naive Bayes Results:")
print(f"Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"                 Negative  Positive")
print(f"Actual Negative     {cm[0,0]:2d}       {cm[0,1]:2d}")
print(f"Actual Positive     {cm[1,0]:2d}       {cm[1,1]:2d}")

true_negatives = cm[0,0]
false_positives = cm[0,1]
false_negatives = cm[1,0]
true_positives = cm[1,1]

precision_pos = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall_pos = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
precision_neg = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
recall_neg = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

print(f"\nDetailed Metrics:")
print(f"Positive Class - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}")
print(f"Negative Class - Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}")

print(f"\nSample Predictions:")
print(f"{'Actual':<10} {'Predicted':<10} {'Review (first 50 chars)'}")
print("-" * 70)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    review_text = df_test.iloc[i]['Review'][:50] + "..."
    status = "✓" if actual == predicted else "✗"
    print(f"{actual:<10} {predicted:<10} {status} {review_text}")

print(f"\nTop 10 Most Informative Features:")
feature_names = vectorizer.get_feature_names_out()
log_probs = nb_classifier.feature_log_prob_

pos_features = np.argsort(log_probs[1])[::-1][:10]
print(f"\nTop words for POSITIVE sentiment:")
for i, feature_idx in enumerate(pos_features, 1):
    word = feature_names[feature_idx]
    prob = np.exp(log_probs[1][feature_idx])
    print(f"{i:2d}. {word:<15} (log prob: {log_probs[1][feature_idx]:.4f})")

neg_features = np.argsort(log_probs[0])[::-1][:10]
print(f"\nTop words for NEGATIVE sentiment:")
for i, feature_idx in enumerate(neg_features, 1):
    word = feature_names[feature_idx]
    prob = np.exp(log_probs[0][feature_idx])
    print(f"{i:2d}. {word:<15} (log prob: {log_probs[0][feature_idx]:.4f})")

print(f"\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"✓ Dataset: 100 movie reviews (50 positive, 50 negative)")
print(f"✓ Tokenization: 500 features, stop words removed")
print(f"✓ Train-Test Split: 80-20 stratified split")
print(f"✓ Model: Multinomial Naive Bayes")
print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✓ Training samples: {X_train.shape[0]}")
print(f"✓ Testing samples: {X_test.shape[0]}")
print(f"✓ Feature matrix shape: {X_train.shape[1]} features")

print(f"\n" + "=" * 70)
print("CUSTOM PREDICTION FUNCTION")
print("=" * 70)

def predict_review_sentiment(model, vectorizer, review):
    """
    Predict the sentiment of a single movie review.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained sentiment classification model
    vectorizer : CountVectorizer
        Fitted CountVectorizer used for tokenization
    review : str
        A single movie review text to classify
    
    Returns:
    --------
    str
        Predicted sentiment ('positive' or 'negative')
    """
    review_vectorized = vectorizer.transform([review])
    
    prediction = model.predict(review_vectorized)
    
    return prediction[0]

print("Testing the predict_review_sentiment function:")
print("=" * 50)

test_reviews = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "Terrible film with bad acting and boring plot. Complete waste of time.",
    "The cinematography was beautiful and the story was compelling.",
    "I fell asleep halfway through. Very disappointing and poorly made.",
    "Outstanding performances and brilliant direction. Highly recommended!",
    "The worst movie I've ever seen. Awful script and terrible acting."
]

print("Sample predictions using the trained model:")
print(f"{'Review':<60} {'Predicted Sentiment'}")
print("-" * 80)

for i, review in enumerate(test_reviews, 1):
    predicted_sentiment = predict_review_sentiment(nb_classifier, vectorizer, review)
    review_short = review[:55] + "..." if len(review) > 55 else review
    print(f"{review_short:<60} {predicted_sentiment}")

print(f"\nTesting edge cases:")
print("-" * 40)

edge_cases = [
    "Okay movie, nothing special.",
    "Good and bad moments throughout.",
    "The movie.",
    "Amazing terrible good bad excellent awful."
]

for review in edge_cases:
    predicted_sentiment = predict_review_sentiment(nb_classifier, vectorizer, review)
    print(f"'{review}' -> {predicted_sentiment}")

print(f"\nPrediction probabilities for sample reviews:")
print("-" * 50)

for i, review in enumerate(test_reviews[:3], 1):
    review_vectorized = vectorizer.transform([review])
    probabilities = nb_classifier.predict_proba(review_vectorized)[0]
    predicted_class = nb_classifier.classes_[np.argmax(probabilities)]
    
    print(f"\nReview {i}: '{review[:40]}...'")
    print(f"Negative probability: {probabilities[0]:.4f}")
    print(f"Positive probability: {probabilities[1]:.4f}")
    print(f"Predicted: {predicted_class}")

print(f"\n" + "=" * 50)
print("FUNCTION USAGE EXAMPLE")
print("=" * 50)
print("# Example usage of the function:")
print("# prediction = predict_review_sentiment(nb_classifier, vectorizer, 'Your review here')")
print("# print(f'Predicted sentiment: {prediction}')")

print(f"\n✓ Function predict_review_sentiment() is ready to use!")
print(f"✓ Takes trained model, fitted vectorizer, and review string as input")
print(f"✓ Returns predicted sentiment as string ('positive' or 'negative')")
print(f"✓ Tested with sample reviews and edge cases")