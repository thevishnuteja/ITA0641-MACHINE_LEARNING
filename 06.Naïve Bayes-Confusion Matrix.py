import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Preparation
data = {'text': ['I love this movie', 'This movie is terrible', 'I enjoyed this film', 'I hate this movie', 'Fantastic film', 'Worst film ever'],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']}
df = pd.DataFrame(data)

# 2. Feature Extraction
X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Model Prediction
y_pred = model.predict(X_test)

# 5. Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

# Output results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
