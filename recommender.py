import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Function to get movie recommendations based on the cosine similarity scores of movie descriptions.
def get_recommendations(data_frame, title, cosine_similarity, indices):
    # Get the index of the movie that matches the given title.
    idx = indices[title]

    # Compute the pairwise cosine similarity scores for all movies with the given movie.
    sim_scores = list(enumerate(cosine_similarity[idx]))

    # Sort the movies based on the similarity scores in descending order.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies, excluding the first one as it is the movie itself.
    sim_scores = sim_scores[1:11]

    # Extract the movie indices for the top 10 similar movies.
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies.
    return data_frame['title'].iloc[movie_indices]


# Function to calculate the similarity matrix and return movie recommendations.
def similarity(data_frame, title):
    # Initialize a TF-IDF Vectorizer, removing all english stop words.
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN values in 'overview' column with empty strings.
    data_frame['overview'] = data_frame['overview'].fillna('')

    # Transform the 'overview' text to a matrix of TF-IDF features.
    tfidf_matrix = tfidf.fit_transform(data_frame['overview'])

    # Compute the cosine similarity matrix from the TF-IDF matrix.
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a Series from the DataFrame indices with the movie title as the index (key).
    indices = pd.Series(data_frame.index, index=data_frame['title']).drop_duplicates()


    # Return the top 10 most similar movies for the given title.
    return get_recommendations(data_frame=data_frame, title=title, cosine_similarity=cosine_sim, indices=indices)

