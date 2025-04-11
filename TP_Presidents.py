import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# === CONFIGURATION ===
nltk.data.path = ['/usr/local/nltk_data']
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

def get_pos_tags(text):
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)

# === SCRIPT PRINCIPAL ===

def main():
    # Chargement des données
    df = pd.read_csv("inaug_speeches.csv", encoding="ISO-8859-1")
    print("Colonnes disponibles :", df.columns.tolist())
    df.drop_duplicates(subset='text', inplace=True)

    # Initialisation
    global stop_words, lemmatizer, stemmer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Nettoyage et analyse
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['stemmed_text'] = df['text'].apply(stem_text)
    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    # POS tagging sur texte brut (meilleure qualité)
    df['pos_tags'] = df['text'].apply(get_pos_tags)
    print("\n📌 Exemple de POS tags pour un discours :")
    print(df['pos_tags'].iloc[0][:15])

    # Histogramme des sentiments
    plt.figure(figsize=(8, 4))
    plt.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution des sentiments des discours")
    plt.xlabel("Score de sentiment")
    plt.ylabel("Nombre de discours")
    plt.show()

    # Nuage de mots
    text = " ".join(df['cleaned_text'])
    combined_stopwords = stop_words.union(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, stopwords=combined_stopwords).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nuage de mots des discours")
    plt.show()

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['cleaned_text'])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    print("\n📋 Exemple de matrice TF-IDF :")
    print(tfidf_df.head())

    # Barplot des top TF-IDF
    mean_tfidf = tfidf_df.mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    mean_tfidf.plot(kind='bar', color='darkorange')
    plt.title("Top 20 mots avec les scores TF-IDF les plus élevés")
    plt.xlabel("Mot")
    plt.ylabel("Score TF-IDF moyen")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Analyse par président
    if 'president' in df.columns:
        sentiment_by_president = df.groupby('president')['sentiment'].mean().sort_values()
        print("\n📌 Sentiment moyen par président :")
        print(sentiment_by_president)

        plt.figure(figsize=(12, 6))
        sentiment_by_president.plot(kind='barh', color='steelblue')
        plt.title("Sentiment moyen par président")
        plt.xlabel("Score de sentiment moyen")
        plt.ylabel("Président")
        plt.tight_layout()
        plt.show()

    # Évolution du sentiment dans le temps
    if 'year' in df.columns:
        plt.figure(figsize=(10, 5))
        df.groupby('year')['sentiment'].mean().plot(marker='o', color='green')
        plt.title("Évolution du sentiment moyen par année")
        plt.xlabel("Année")
        plt.ylabel("Sentiment moyen")
        plt.grid(True)
        plt.show()

    # Analyse des POS tags (types de mots)
    all_tags = [tag for sublist in df['pos_tags'] for _, tag in sublist]
    pos_counts = Counter(all_tags)
    print("\n📊 Fréquence des types de mots (POS tags) :")
    for tag, count in pos_counts.most_common(10):
        print(f"{tag} : {count}")

# === LANCEMENT ===
if __name__ == "__main__":
    main()
