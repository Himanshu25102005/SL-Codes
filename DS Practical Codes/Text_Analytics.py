# Generated from: Text_Analytics.ipynb
# Converted at: 2026-04-23T03:18:50.148Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Text Analytics 


# PROBLEM STATEMENT: 
# 1. Extract Sample document and apply following document preprocessing methods: Tokenization, POS 
# Tagging, stop words removal, Stemming and Lemmatization. 
# 2. Create representation of documents by calculating Term Frequency and Inverse Document Frequency. 


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

text = """Text analytics is the process of analyzing textual data. 
It involves tokenization, stop word removal, stemming and lemmatization. 
Machine learning models use this processed data for better predictions."""

sent_tokens = sent_tokenize(text)
word_tokens = word_tokenize(text)

print("Sentence Tokens:", sent_tokens)
print("Word Tokens:", word_tokens)

stop_words = set(stopwords.words('english'))

filtered_words = [w for w in word_tokens if w.lower() not in stop_words]

print("After Stopword Removal:", filtered_words)

pos_tags = pos_tag(filtered_words)

print("POS Tags:", pos_tags)

stemmer = PorterStemmer()

stemmed_words = [stemmer.stem(w) for w in filtered_words]

print("Stemmed Words:", stemmed_words)

lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

print("Lemmatized Words:", lemmatized_words)

documents = [text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())