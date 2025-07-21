# movie_reviews_analysis.py

import nltk
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

# Download required resources
nltk.download('movie_reviews')
nltk.download('stopwords')

# Load movie reviews data
docs = [(movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

# Create DataFrame
df = pd.DataFrame(docs, columns=['review_text', 'label'])
print("Sample Reviews:\n", df.sample(5))

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Show cleaned version of one review
original_review = df['review_text'][0]
cleaned_review = preprocess(original_review)
print("\nOriginal Review (first 500 chars):\n", original_review[:500])
print("\nCleaned Review (first 500 chars):\n", cleaned_review[:500])

# Bag of Words
vectorizer = CountVectorizer(max_features=50)
X_bow = vectorizer.fit_transform(df['review_text'])
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print("\nBag of Words Matrix:\n", bow_df.head())

# Top 10 most frequent words
word_counts = bow_df.sum().sort_values(ascending=False)
top_10_words = word_counts.head(10)
print("\nTop 10 Most Frequent Words:\n", top_10_words)

print("""
BoW Frequent Word Explanation:
Words like 'film', 'movie', 'good' likely show up in both positive and negative reviews.
High frequency doesn't always mean sentiment-specific, but it reflects core vocabulary in this domain.
""")

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=50)
X_tfidf = tfidf_vectorizer.fit_transform(df['review_text'])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n", tfidf_df.head())

# Top 5 words in first review
first_review_scores = pd.Series(X_tfidf[0].toarray()[0], index=tfidf_vectorizer.get_feature_names_out())
top_5_tfidf = first_review_scores.sort_values(ascending=False).head(5)
print("\nTop 5 TF-IDF Words in First Review:\n", top_5_tfidf)

print("""
TF-IDF Explanation:
Unlike BoW, TF-IDF reduces the influence of frequent/common words and emphasizes unique words to each review.
This helps improve performance in tasks like sentiment classification by highlighting distinctive terms.
""")

# Visualization: Bar Chart
plt.figure(figsize=(10, 5))
top_10_words.plot(kind='bar', title='Top 10 Most Frequent Words (BoW)', color='skyblue')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization: Word Cloud
combined_text = " ".join(df['review_text'].tolist())
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(combined_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Movie Reviews")
plt.show()
