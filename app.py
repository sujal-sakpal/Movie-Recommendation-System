from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movies_data = pd.read_csv('datasets/movies.csv')
    selected_features = ['genres','keywords','tagline','cast','director']
    for feature in selected_features:
         movies_data[feature] = movies_data[feature].fillna('')
    combined_features = movies_data['genres'] + movies_data['keywords'] + movies_data['tagline'] + movies_data['cast'] + movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    movie_name = request.form['movie_name']
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if title_from_index.lower() != movie_name.lower() and i < 31:
            recommendations.append(title_from_index)
            i += 1

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
