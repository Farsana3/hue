import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Apple is looking at buying U.K. startup for $1 billion. The deal was finalized in September, and everyone was excited about the acquisition."

# Process the text
doc = nlp(text)

# 🔹 1. Tokenization: break into tokens
tokens = [token.text for token in doc]
print("Tokens:")
print(tokens)

# 🔹 2. Remove stop words
tokens_no_stop = [token.text for token in doc if not token.is_stop]
print("\nTokens without stop words:")
print(tokens_no_stop)

# 🔹 3. Lemmatization: show original and lemma
print("\nLemmatization (Original -> Lemma):")
for token in doc:
    print(f"{token.text} -> {token.lemma_}")
