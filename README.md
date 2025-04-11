# 🗽 Analyse NLP des discours d’investiture des présidents américains

Ce projet propose une analyse des discours d’investiture des présidents des États-Unis à l’aide de techniques de **Traitement Automatique du Langage Naturel (NLP)** et de **visualisation de données**.

L’objectif est d’identifier les émotions exprimées, les thèmes dominants, ainsi que les mots les plus significatifs dans le temps, à travers un corpus historique riche.

---

##� Objectif

Appliquer différentes techniques de NLP sur un corpus de discours présidentiels afin d’en extraire des tendances sémantiques, des tonalités, et des indices linguistiques pertinents, tout en illustrant les résultats via des visualisations claires.

---

## Jeu de données

- **Source :** [Kaggle - Presidential Inaugural Addresses Dataset](https://www.kaggle.com/datasets/adhok93/presidentialaddress)
- **Fichier utilisé :** `inaug_speeches.csv`
- Le fichier contient le texte complet de chaque discours, ainsi que le nom du président et l’année.

---

## Méthodes utilisées

- **Tokenisation**
- **Lemmatisation et stemming**
- **Analyse de sentiment** (avec TextBlob)
- **Vectorisation TF-IDF**
- **Suppression des stopwords** (personnalisée avec NLTK et WordCloud)

---

## Visualisations produites

- Histogramme de la polarité des sentiments
- Nuage de mots des termes les plus fréquents
- Graphique en barres des scores TF-IDF les plus élevés
- Courbe de l’évolution du sentiment dans le temps
- Moyenne du sentiment par président

---

## Bonnes pratiques appliquées

- Code Python propre, modulaire et structuré par fonctions
- Nettoyage rigoureux des données (suppression doublons, normalisation)
- Visualisations lisibles avec titres, axes et couleurs soignées
- Rapport clair avec interprétation des résultats


## 👨‍💻 Auteur

**Léo** – Étudiant en informatique  
Projet réalisé dans le cadre d’un TP sur la data visualisation et le traitement de texte (avril 2025)

