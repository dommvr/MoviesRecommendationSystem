import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
import os

ps = PorterStemmer()
stopwords = stopwords.words('english')

def stem(text):
    stem_text = []
    for word in text.split():
        stem_text.append(ps.stem(word))
    text = ' '.join(stem_text)

    return text

def clear_text(text):
    text = str(text)
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = stem(text)

    return text

current_directory = os.getcwd()
file = os.path.join(current_directory, 'data', 'movies.csv')
movies_df = pd.read_csv(file)
movies_df['tags'] = movies_df['tags'].apply(clear_text)
vectors = CountVectorizer().fit_transform(movies_df['tags']).toarray()
movies_similarity = cosine_similarity(vectors)

while True:
    title = input('Movie title: ')
    if title in movies_df['title'].values:
        break
    else:
        print('Wrong movie title')

movie_index = np.where(movies_df['title'] == title)[0][0]
similar_movies = sorted(list(enumerate(movies_similarity[movie_index])), reverse=True, key=lambda x:x[1])

print('Recommended movies: ')
for movie in similar_movies[1:6]:
    print(movies_df.iloc[movie[0]]['title'])