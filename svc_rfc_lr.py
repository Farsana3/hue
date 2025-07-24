import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import copy

nltk.download('movie_reviews')

# Load and prepare data
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

np.random.seed(42)
np.random.shuffle(documents)

texts = [text for text, label in documents]
labels = [1 if label == 'pos' else 0 for text, label in documents]

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(kernel='linear', probability=True)
}

# --- Evaluate Each Model ---
results = {}

for name, model in models.items():
    print(f"\n🔍 Evaluating: {name}")
    
    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_accuracies = []
    best_accuracy = 0
    best_model = None

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        kf_accuracies.append(acc)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = copy.deepcopy(model)

    mean_kf_acc = np.mean(kf_accuracies)

    # --- Holdout Validation ---
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    
    model.fit(X_train_h, y_train_h)
    holdout_acc = model.score(X_test_h, y_test_h)

    # --- Save Results ---
    results[name] = {
        "KFold Mean": mean_kf_acc,
        "KFold Best": best_accuracy,
        "Holdout": holdout_acc
    }

# --- Print Summary ---
print("\n📊 Model Comparison Summary:")
print("{:<20} {:>15} {:>15} {:>15}".format("Model", "KFold Mean", "KFold Best", "Holdout"))
print("-" * 65)
for name, scores in results.items():
    print("{:<20} {:>15.4f} {:>15.4f} {:>15.4f}".format(
        name, scores["KFold Mean"], scores["KFold Best"], scores["Holdout"]
    ))

# --- Identify Best Model Overall (by K-Fold Mean) ---
best_model_name = max(results.items(), key=lambda x: x[1]["KFold Mean"])[0]
print(f"\n🏆 Best Performing Model (by K-Fold Mean Accuracy): **{best_model_name}**")
