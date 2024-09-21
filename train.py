from utils.utils import train_and_save_model, load_data

if __name__ == "__main__":
    file_name = 'ml-100k'
    file_path = ''.join(['./', file_name])
    data = load_data()

    movie_id_to_name = {}
    movie_name_to_id = {}

    with open(''.join([file_path, '/u.item']), 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('|')
            movie_id = int(parts[0])
            movie_title = parts[1]
            movie_id_to_name[movie_id] = movie_title
            movie_name_to_id[movie_title.lower()] = movie_id

    train_and_save_model(data)