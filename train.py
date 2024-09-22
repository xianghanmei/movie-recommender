from utils.utils import train_and_save_model, load_data

if __name__ == "__main__":
    # Load the data
    data = load_data()

    # Construct the mapping dictionaries between movie names and raw ids
    movie_id_to_name = {}
    movie_name_to_id = {}

    file_name = 'ml-100k'
    file_path = ''.join(['./', file_name])

    with open(''.join([file_path, '/u.item']), 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('|')
            movie_id = int(parts[0])
            movie_title = parts[1]
            movie_id_to_name[movie_id] = movie_title
            movie_name_to_id[movie_title.lower()] = movie_id

    # Train and save the model
    train_and_save_model(data, save_path='./model/knn_movie_recommender.model')