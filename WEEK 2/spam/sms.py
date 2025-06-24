import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gensim
from gensim.models import Word2Vec
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df.rename(columns={'v1': 'Label', 'v2': 'Message'})
df = df[['Label', 'Message']]  

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['Label'].value_counts()}")

def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation
    - Tokenizing
    - Removing stop words
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    return tokens
print("Preprocessing messages...")
df['processed_tokens'] = df['Message'].apply(preprocess_text)

all_tokens = df['processed_tokens'].tolist()

print("Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences=all_tokens,
    vector_size=100, 
    window=5,       
    min_count=2,      
    workers=4,      
    sg=0      
)

def message_to_vector(tokens, model, vector_size=100):
    """
    Convert a tokenized message to a fixed-length vector by averaging
    the Word2Vec vectors of all words in the message
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

print("Converting messages to vectors...")
X = np.array([message_to_vector(tokens, w2v_model) for tokens in df['processed_tokens']])
y = df['Label'].map({'ham': 0, 'spam': 1}).values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print("Training Logistic Regression classifier...")
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
def predict_message_class(model, w2v_model, message):
    """
    Predict the class of a single message
    
    Args:
        model: Trained logistic regression model
        w2v_model: Trained Word2Vec model
        message: Input message string
    
    Returns:
        Predicted class ('spam' or 'ham')
    """

    tokens = preprocess_text(message)
    vector = message_to_vector(tokens, w2v_model).reshape(1, -1)

    prediction = model.predict(vector)[0]

    return 'spam' if prediction == 1 else 'ham'
print("\n" + "="*50)
print("Testing predict_message_class function:")
print("="*50)

test_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hey, are you free for lunch tomorrow?",
    "URGENT: Your account will be suspended. Reply with your password immediately.",
    "Thanks for the great meeting today. Looking forward to our next discussion."
]

for msg in test_messages:
    prediction = predict_message_class(classifier, w2v_model, msg)
    print(f"Message: '{msg}'")
    print(f"Prediction: {prediction}")
    print("-" * 50)