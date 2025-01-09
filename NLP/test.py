# Importation des bibliothèques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Télécharger les données nécessaires
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


stop_words = set(stopwords.words('french'))

# Charger les données
file_path = "../tweets_labelled.csv"
df = pd.read_csv(file_path)

# Supprimer les lien et les emojis
def remove_links(text):
    text = re.sub(r"http\S+", "", text)  # Supprimer les liens
    return text

df['cleaned_text'] = df['Texte'].astype(str).apply(remove_links)

# Fonction pour extraire les mentions sans modifier le reste du texte
def get_mentions(text):
    mentions = re.findall(r"@\S+", text)  # Extraire les mentions
    return mentions

# Fonction pour supprimer les mentions sans affecter les accents
def remove_mentions(text):
    # Utiliser re.sub pour supprimer uniquement les mentions
    return re.sub(r"@\S+", "", text)

# Appliquer sur les données
df['mentions'] = df['cleaned_text'].apply(get_mentions)  # Extraire les mentions
df['cleaned_text'] = df['cleaned_text'].apply(remove_mentions)  # Supprimer les mentions

from collections import Counter
import re

# Fonction pour extraire et compter les caractères spéciaux
def special_characters_count(text):
    # Trouver tous les caractères spéciaux
    special_chars = re.findall(r"[^a-zA-Z0-9\s]", text)
    # Retourner un dictionnaire des occurrences
    return Counter(special_chars)

# Appliquer sur la colonne 'cleaned_text' et agréger les résultats
special_characters_total = df['cleaned_text'].apply(special_characters_count)

# Fusionner tous les comptes en un seul Counter
total_count = Counter()
for count in special_characters_total:
    total_count.update(count)

# Supprimer les caractères spéciaux
def remove_special_characters(text):
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text) # Supprimer les caractères spéciaux
    return text

df['cleaned_text'] = df['cleaned_text'].apply(remove_special_characters)

# Supprimer la colonne Media
df.drop(columns=['Media'], inplace=True)

# Si une ligne a un index ou sujet est manquant, supprimez-le
df.dropna(subset=['Index'], inplace=True)
df.dropna(subset=['Sujet'], inplace=True)

# Supprimer les mots inutiles
stop_words = set(stopwords.words('french'))

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(text)


df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

# Tokeniser les textes nettoyés
df['tokens'] = df['cleaned_text'].apply(lambda x: word_tokenize(x.lower()))

# Entraîner le modèle Word2Vec
sentences = df['tokens'].tolist()  # Obtenir toutes les phrases tokenisées
model = Word2Vec(sentences, vector_size=200, window=5, min_count=1, workers=4)

# Mots clés pour détecter les tweets sur les voitures électriques
keywords = [
    "électricité", "voiture", "voitures", "véhicules", "véhicule", 
    "tesla", "recharge", "batterie", "batteries", "thermique", 
    "thermiques", "hybride", "hybrides", "bornes", "électrique", 
    "électriques"
]

# Fonction pour calculer la similarité moyenne entre les mots d'un tweet et les mots-clés
def calculate_similarity(tweet_tokens, model, keywords):
    similarities = []
    for word in tweet_tokens:
        if word in model.wv:  # Vérifiez si le mot est dans le vocabulaire du modèle
            word_similarities = [model.wv.similarity(word, keyword) for keyword in keywords if keyword in model.wv]
            if word_similarities:
                similarities.append(max(word_similarities))
    return np.mean(similarities) if similarities else 0

# Calculer la similarité pour chaque tweet
df['similarity'] = df['tokens'].apply(lambda x: calculate_similarity(x, model, keywords))

# Filtrer les tweets pertinents (similarité > seuil, par exemple 0.5)
relevant_tweets = df[df['similarity'] > 0.6]

# Random Forest Classifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Diviser les données en ensembles d'entraînement et de test
X = df['cleaned_text']
y = df['Sujet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un vecteur TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraîner un classificateur RandomForest
clf = RandomForestClassifier()
clf.fit(X_train_tfidf, y_train)

# Prédire les catégories
y_pred = clf.predict(X_test_tfidf)

# Afficher les résultats
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Diviser les données en ensembles d'entraînement et de test
X = df['cleaned_text'].tolist()
y = df['Sujet'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger le tokenizer et le modèle
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(df['Sujet'].unique())
)

# Fonction pour tokeniser les données
def tokenize_data(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
    )

# Préparer les données pour Hugging Face Dataset
train_data = Dataset.from_dict({'text': X_train, 'label': y_train})
test_data = Dataset.from_dict({'text': X_test, 'label': y_test})

# Tokeniser les données
train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)

# Définir les colonnes nécessaires pour Trainer
train_data = train_data.rename_column("label", "labels")
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

test_data = test_data.rename_column("label", "labels")
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Créer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',  # Dossier pour sauvegarder les checkpoints
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',  # Dossier pour les fichiers de logs
    logging_steps=10,
    evaluation_strategy='epoch',
)

# Créer un objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Entraîner le modèle
trainer.train()

# Évaluer le modèle
predictions = trainer.predict(test_data)
y_pred = np.argmax(predictions.predictions, axis=1)

# Afficher les résultats
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
