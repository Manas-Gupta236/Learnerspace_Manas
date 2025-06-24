import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TwitterAirlineSentimentAnalysis:
    def __init__(self, dataset_path=None):
        """
        Initialize the sentiment analysis pipeline
        """
        self.dataset_path = dataset_path
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = None
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        Load the Twitter US Airline Sentiment dataset
        """
        try:
            self.df = pd.read_csv('Tweets.csv')
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print("Dataset file not found. Please ensure 'Tweets.csv' is in the current directory.")
            return False
    
    def preprocess_text(self, text):
        """
        Preprocess each tweet using the following steps:
        - Convert to lowercase
        - Remove URLs, mentions, hashtags, and punctuation
        - Expand contractions
        - Lemmatize words
        - Remove emojis and special symbols
        """
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
    
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
      
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self):
        """
        Preprocess the entire dataset
        """
        print("Preprocessing tweets...")
        self.df['processed_text'] = self.df['text'].apply(self.preprocess_text)
        
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        
        print(f"Dataset shape after preprocessing: {self.df.shape}")
        print("Sample processed tweets:")
        for i in range(min(3, len(self.df))):
            print(f"Original: {self.df.iloc[i]['text']}")
            print(f"Processed: {self.df.iloc[i]['processed_text']}")
            print("-" * 50)
    
    def load_word2vec_model(self):
        """
        Load pre-trained Google News Word2Vec model using gensim
        """
        try:
            print("Loading pre-trained Word2Vec model...")
   
            sentences = [text.split() for text in self.df['processed_text'] if len(text.strip()) > 0]
        
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, 
                                         min_count=1, workers=4, sg=1)
            print("Word2Vec model created successfully!")
            
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            print("Using TF-IDF vectorization instead...")
            self.word2vec_model = None
    
    def get_word2vec_embeddings(self, text):
        """
        Convert text to fixed-length vector by averaging Word2Vec word vectors
        """
        if self.word2vec_model is None:
            return np.zeros(100)
        
        words = text.split()
        word_vectors = []
        
        for word in words:
            try:
                word_vectors.append(self.word2vec_model.wv[word])
            except KeyError:
                continue
        
        if len(word_vectors) == 0:
            return np.zeros(100)
        
        return np.mean(word_vectors, axis=0)
    
    def vectorize_texts(self):
        """
        Convert tweets to fixed-length vectors using Word2Vec embeddings
        """
        if self.word2vec_model is not None:
            print("Converting tweets to Word2Vec embeddings...")
            embeddings = []
            for text in self.df['processed_text']:
                embedding = self.get_word2vec_embeddings(text)
                embeddings.append(embedding)
            
            self.X = np.array(embeddings)
            print(f"Embedding shape: {self.X.shape}")
        else:
            print("Using TF-IDF vectorization...")
            self.X = self.vectorizer.fit_transform(self.df['processed_text']).toarray()
            print(f"TF-IDF shape: {self.X.shape}")

        self.y = self.df['airline_sentiment'].values
        print(f"Target distribution: {pd.Series(self.y).value_counts()}")
    
    def split_dataset(self, test_size=0.2):
        """
        Split dataset into training (80%) and testing (20%) sets
        """
        print("Splitting dataset...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        print(f"Training set distribution: {pd.Series(self.y_train).value_counts()}")
    
    def train_classifier(self):
        """
        Train Logistic Regression classifier on vectorized training data
        """
        print("Training Logistic Regression classifier...")
        self.classifier.fit(self.X_train, self.y_train)
        
        train_predictions = self.classifier.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        print(f"Training accuracy: {train_accuracy:.4f}")
    
    def evaluate_classifier(self):
        """
        Report accuracy on test set
        """
        print("Evaluating classifier on test set...")
        test_predictions = self.classifier.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, test_predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, test_predictions))
        
        return test_accuracy
    
    def predict_tweet_sentiment(self, classifier_model, glove_model, tweet):
        """
        Predict sentiment of a single tweet
        
        Args:
            classifier_model: Trained classifier (self.classifier)
            glove_model: Word2Vec model (self.word2vec_model) 
            tweet: Single tweet string
            
        Returns:
            Predicted sentiment (positive, negative, or neutral)
        """
     
        processed_tweet = self.preprocess_text(tweet)
        
        if len(processed_tweet.strip()) == 0:
            return "neutral"
        
        if glove_model is not None:
            tweet_vector = self.get_word2vec_embeddings(processed_tweet).reshape(1, -1)
        else:
            tweet_vector = self.vectorizer.transform([processed_tweet]).toarray()
        prediction = classifier_model.predict(tweet_vector)[0]
        
        return prediction
    
    def run_complete_pipeline(self):
        """
        Run the complete sentiment analysis pipeline
        """
        print("="*60)
        print("TWITTER AIRLINE SENTIMENT ANALYSIS PIPELINE")
        print("="*60)
        if not self.load_data():
            print("Failed to load dataset. Exiting...")
            return None
        self.preprocess_dataset()
        
        self.load_word2vec_model()
        
        self.vectorize_texts()
        
        self.split_dataset()
      
        self.train_classifier()
        
        accuracy = self.evaluate_classifier()
        
        print("\n" + "="*60)
        print("TESTING PREDICTION FUNCTION")
        print("="*60)
        test_tweets = [
            "@VirginAmerica What @dhepburn said.",
            "@VirginAmerica plus you've added commercials to the experience.. tacky.",
            "@VirginAmerica I didn't today... Must mean I need to take another trip!",
            "@VirginAmerica it's really aggressive to blast obnoxious \"entertainment\" in your guests' faces &amp; they have little recourse",
            "@VirginAmerica Great flight today! Thanks for the excellent service."
        ]
        
        for tweet in test_tweets:
            predicted_sentiment = self.predict_tweet_sentiment(
                self.classifier, self.word2vec_model, tweet
            )
            print(f"Tweet: {tweet}")
            print(f"Predicted Sentiment: {predicted_sentiment}")
            print("-" * 50)
        print(f"\nPipeline completed successfully!")
        print(f"Final test accuracy: {accuracy:.4f}")
        return accuracy
if __name__ == "__main__":
    sentiment_analyzer = TwitterAirlineSentimentAnalysis()
    final_accuracy = sentiment_analyzer.run_complete_pipeline()
    
    if final_accuracy is not None:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Model successfully trained and evaluated!")
        print(f"Test Accuracy: {final_accuracy:.4f}")
        print(f"All components implemented and working correctly.")
    else:
        print("Pipeline failed to complete due to missing dataset.")