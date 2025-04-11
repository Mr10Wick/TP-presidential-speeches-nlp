# Importation de mon fichier CSV depuis mon local 

# from google.colab import files
# uploaded = files.upload()

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# === CONFIGURATION ===
nltk.data.path = ['/usr/local/nltk_data']

# === FONCTIONS ===

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized)

def stem_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(stemmed)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# === CHARGEMENT DES DONNÉES ===

df = pd.read_csv("inaug_speeches.csv", encoding="ISO-8859-1")
print("Colonnes disponibles :", df.columns.tolist())
df.drop_duplicates(subset='text', inplace=True)

# === PRÉTRAITEMENT ===

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

df['cleaned_text'] = df['text'].apply(clean_text)
df['stemmed_text'] = df['text'].apply(stem_text)
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# === HISTOGRAMME DE SENTIMENT ===

plt.figure(figsize=(8, 4))
plt.hist(df['sentiment'], bins=20, color='cornflowerblue', edgecolor='black')
plt.title("Distribution des sentiments des discours")
plt.xlabel("Score de sentiment")
plt.ylabel("Nombre de discours")
plt.show()

# === NUAGE DE MOTS ===

text = " ".join(df['cleaned_text'])
combined_stopwords = stop_words.union(STOPWORDS)
wordcloud = WordCloud(width=800, height=400, stopwords=combined_stopwords).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nuage de mots des discours")
plt.show()

# === TF-IDF ===

vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['cleaned_text'])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("\n📋 Exemple de matrice TF-IDF :")
print(tfidf_df.head())

# === BARPLOT TF-IDF ===

mean_tfidf = tfidf_df.mean().sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 6))
mean_tfidf.plot(kind='bar', color='darkorange')
plt.title("Top 20 mots (score moyen TF-IDF)")
plt.xlabel("Mot")
plt.ylabel("Score TF-IDF")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === ANALYSE PAR PRÉSIDENT ===

if 'president' in df.columns:
    sentiment_by_president = df.groupby('president')['sentiment'].mean().sort_values()
    
    print("\n📌 Sentiment moyen par président :")
    print(sentiment_by_president)
    
    # 👉 VISUALISATION ajoutée ici
    plt.figure(figsize=(12, 6))
    sentiment_by_president.plot(kind='barh', color='steelblue')
    plt.title("Sentiment moyen par président")
    plt.xlabel("Score de sentiment moyen")
    plt.ylabel("Président")
    plt.tight_layout()
    plt.show()

# === ÉVOLUTION DANS LE TEMPS ===

if 'year' in df.columns:
    plt.figure(figsize=(10, 5))
    df.groupby('year')['sentiment'].mean().plot(marker='o', color='green')
    plt.title("Évolution du sentiment moyen par année")
    plt.xlabel("Année")
    plt.ylabel("Sentiment moyen")
    plt.grid(True)
    plt.show()
