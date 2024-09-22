import streamlit as st
from utils.utils import load_data, load_model_and_recommend, get_movie_name, get_filtered_movies

if __name__ == "__main__":
    # Load the data
    file_name = 'ml-100k'
    file_path = ''.join(['./', file_name])
    data = load_data(file_path)

    # Construct the mapping dictionaries between movie names and raw ids
    movie_id_to_name = {}
    movie_name_to_id = {}

    with open(''.join([file_path, '/u.item']), 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('|')
            movie_id = int(parts[0])
            movie_title = parts[1]
            movie_id_to_name[movie_id] = movie_title
            movie_name_to_id[movie_title.lower()] = movie_id

    # Extract movies used in the training process
    model_path = './model/knn_movie_recommender.model'
    filtered_movies = get_filtered_movies(model_path, movie_id_to_name)

    # Streamlit App title
    st.title("Movie Recommender")

    # Set up a drop-down menu for movie selection to avoid movie name mismatch
    movie_options = filtered_movies
    selected_movie_name = st.selectbox(
        "Select a movie you like:",
        options=movie_options
    )

    # Set up a button to trigger recommendation
    if st.button("Recommend"):
        recommendation = load_model_and_recommend(selected_movie_name, 
                                                model_path, 
                                                movie_name_to_id=movie_name_to_id, 
                                                movie_id_to_name=movie_id_to_name)
        st.write(f"If you liked '{selected_movie_name}', you might also like '{recommendation}' according to our prediction.")