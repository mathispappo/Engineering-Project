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
    text = text.encode('ascii', 'ignore').decode('ascii') # supprimer les emojis
    text = text.lower()  # Convertir en minuscules
    return text

df['cleaned_text'] = df['Texte'].astype(str).apply(remove_links)

# Mettre les mentions dans une autre colonne et les supprimer du texte
def get_mentions(text):
    mentions = re.findall(r"@\S+", text)
    return mentions

def remove_mentions(text):
    text = re.sub(r"@\S+", "", text)
    return text

df['mentions'] = df['cleaned_text'].apply(get_mentions)
df['cleaned_text'] = df['cleaned_text'].apply(remove_mentions)

# visualiser uniquement les caractères spéciaux
def special_characters(text):
    text = re.sub(r"[a-zA-Z\s]", "", text)  # Supprimer les caractères spéciaux
    return text

# afficher la totalité des caractères spéciaux
print(df['cleaned_text'].apply(special_characters).sum())

# afficher uniquement les mentions
def mentions(text):
    text = re.findall(r"@\S+", text)  # Trouver les mentions
    return text

# afficher les mentions
print(df['cleaned_text'].apply(mentions).sum())

# Supprimer les caractères spéciaux
def remove_special_characters(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Supprimer les caractères spéciaux
    return text

df['cleaned_text'] = df['cleaned_text'].apply(remove_special_characters)

# des données manquantes
print(df.isnull().sum())

# Supprimer la colonne Media
df.drop(columns=['Media'], inplace=True)

# Si une ligne a un index manquant, supprimez-le
df.dropna(subset=['Index'], inplace=True)

# des données manquantes
print(df.isnull().sum())

# Supprimer les mots inutiles
stop_words = set(stopwords.words('french'))

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(text)


df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

# Analyse des catégories (distribution des sujets)
plt.figure(figsize=(8, 5))
sns.countplot(y='Sujet', data=df, order=df['Sujet'].value_counts().index)
plt.title("Distribution des Sujets")
plt.show()

# Analyse des longueurs de texte
df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(df['text_length'], bins=20, kde=True)
plt.title("Distribution des longueurs de texte")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de textes")
plt.show()

# Nuage de mots (WordCloud)
all_text = " ".join(df['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nuage de mots des textes")
plt.show()

# Fréquences des mots
from collections import Counter

all_words = " ".join(df['cleaned_text']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(10)

# Visualiser les mots les plus fréquents
words, counts = zip(*common_words)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(counts), y=list(words))
plt.title("Top 10 des mots les plus fréquents")
plt.xlabel("Fréquence")
plt.ylabel("Mots")
plt.show()