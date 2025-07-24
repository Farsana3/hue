import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import copy

# Download movie_reviews if not already downloaded
nltk.download('movie_reviews')

# Load and shuffle data
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

np.random.seed(42)
np.random.shuffle(documents)

# Extract text and labels
texts = [text for text, label in documents]
labels = [1 if label == 'pos' else 0 for text, label in documents]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# --- K-Fold Cross Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_accuracies = []
best_accuracy = 0
best_model = None

print("\n=== K-Fold Cross Validation ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    acc = model.score(X_val, y_val)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    
    kf_accuracies.append(acc)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = copy.deepcopy(model)

mean_kf_accuracy = np.mean(kf_accuracies)
print(f"Mean K-Fold Accuracy: {mean_kf_accuracy:.4f}")
print(f"Best K-Fold Accuracy: {best_accuracy:.4f}")

# --- Holdout Validation ---
print("\n=== Holdout Validation ===")
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

holdout_model = LogisticRegression(max_iter=1000)
holdout_model.fit(X_train_h, y_train_h)
holdout_accuracy = holdout_model.score(X_test_h, y_test_h)
print(f"Holdout Accuracy: {holdout_accuracy:.4f}")

# --- Comparison ---
print("\n=== Comparison Summary ===")
print(f"K-Fold Mean Accuracy : {mean_kf_accuracy:.4f}")
print(f"Holdout Accuracy     : {holdout_accuracy:.4f}")

if abs(mean_kf_accuracy - holdout_accuracy) < 0.02:
    print("→ Both methods produce similar accuracy.")
elif holdout_accuracy > mean_kf_accuracy:
    print("→ Holdout gives a slightly higher estimate — might be optimistic.")
else:
    print("→ K-Fold gives a more stable estimate — better for reliability.")
