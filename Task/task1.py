import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Load training data
train_df = pd.read_csv("train_data.txt", sep=":::", engine="python", names=["id", "title", "genre", "plot"])
print(train_df.head())

# Step 2: Preprocess text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define a simple stopword list (no need to download NLTK)
stop_words = ['the', 'is', 'in', 'and', 'to', 'of', 'a', 'an']

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words=stop_words, max_features=10000)

# Use the correct column names from your dataset
X = tfidf.fit_transform(train_df['plot'])   # replace 'text' with actual column name
y = train_df['genre']                       # replace 'label' with actual column name

# Step 3: Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=0))

# Step 6: Predict on test data
test_df = pd.read_csv("test_data.txt", sep=":::", engine="python", names=["id", "title", "plot"])
test_df['plot'] = test_df['plot'].fillna("")
X_test = tfidf.transform(test_df['plot'])   # replace 'plot' with actual column name
test_predictions = model.predict(X_test)
# Save predictions
pd.DataFrame({"id": test_df['id'], "genre": test_predictions}).to_csv("task1_predictions.csv", index=False)
print("Predictions saved to task1_predictions.csv")
