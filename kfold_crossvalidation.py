import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import copy

# Download NLTK movie reviews if not already done
nltk.download('movie_reviews')

# Load movie review file IDs and their categories
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle data for randomness
np.random.seed(42)
np.random.shuffle(documents)

# Separate features and labels
texts = [text for text, label in documents]
labels = [1 if label == 'pos' else 0 for text, label in documents]  # 1=pos, 0=neg

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
best_accuracy = 0
best_model = None

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold + 1} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    acc = model.score(X_val, y_val)
    print(f"Accuracy: {acc:.4f}")
    
    accuracies.append(acc)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = copy.deepcopy(model)

# Final results
mean_accuracy = np.mean(accuracies)
print("\n=== Final Evaluation ===")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Best Fold Accuracy: {best_accuracy:.4f}")
