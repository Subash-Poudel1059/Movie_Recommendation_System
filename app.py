from flask import Flask, render_template, request, jsonify
import pickle
import requests
import pandas as pd

app = Flask(__name__)

# Load the movie data and similarity matrix
movies_dict = pickle.load(open(r'E:\datamining\Movie_Recommendation_System\ML_Model\movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)  

similarity = pickle.load(open(r'E:\datamining\Movie_Recommendation_System\ML_Model\similarity.pkl', 'rb'))

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return None  

def recommend(movie):
    if movie not in movies['title'].values:
        return [], []  

    index = movies[movies['title'] == movie].index[0]
    distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []
    
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        poster_url = fetch_poster(movie_id)
        if poster_url:
            recommended_movie_posters.append(poster_url)
            recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

@app.route('/')
def home():
    return render_template('index.html', movies=movies['title'].values)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    movie_name = data.get("movie")

    recommended_movies, posters = recommend(movie_name)

    if not recommended_movies:
        return jsonify({"error": "Movie not found"}), 404

    return jsonify({"movies": recommended_movies, "posters": posters})

if __name__ == '__main__':
    app.run(debug=True)
