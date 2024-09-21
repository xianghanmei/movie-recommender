from surprise import KNNBaseline, dump, Dataset, Reader

def load_data(file_path='./ml-100k/u.data'):
    """
    Load the data with the data directory path
    """
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

def get_movie_id(movie_name, movie_name_to_id):
    """
    Map the id of the movie from the name given the dictionary of name to id
    """
    movie_name_lower = movie_name.lower()
    if movie_name_lower in movie_name_to_id:
        return movie_name_to_id[movie_name_lower]
    else:
        print("Movie not found.")
        return None

def get_movie_name(movie_id, movie_id_to_name):
    """
    Map the name of the movie from the id given the dictionary of id to name
    """
    if movie_id in movie_id_to_name:
        return movie_id_to_name[movie_id]
    else:
        print("Movie ID not found.")
        return None
    
def get_filtered_movies(model_path, movie_id_to_name):
    """
    Return the list of movie names in the training set
    """
    algo = dump.load(model_path)[1]
    trainset_ids = set([int(algo.trainset.to_raw_iid(i)) for i in algo.trainset.all_items()])
    filtered_movies = []
    for movie_id, movie_name in movie_id_to_name.items():
        if movie_id in trainset_ids:
            filtered_movies.append(movie_name)
    return filtered_movies

def train_and_save_model(data, 
                         sim_rule = 'pearson_baseline', 
                         user_based=False, 
                         shrinkage=100,
                         min_support=3,
                         bsl_method='als', 
                         n_epochs=20, 
                         reg_i=15, 
                         reg_u=15,
                         save_path='./model/knn_movie_recommender.model'):
    """
    Use KNNBaseline to train the recommender model, keyword arguments 
    include the original arguments of the function, reference: https://surprise.readthedocs.io/en/stable/knn_inspired.html
    """
    sim_options = {
        'name': sim_rule,
        'user_based': user_based,
        'shrinkage': shrinkage,
        'min_support': min_support
    }

    bsl_options = {
        'method': bsl_method, 
        'n_epochs': n_epochs, 
        'reg_i': reg_i, 
        'reg_u': reg_u
    }
    
    algo = KNNBaseline(k=40, min_k=5, sim_options=sim_options, bsl_options=bsl_options)

    trainset = data.build_full_trainset()
    algo.fit(trainset)

    dump.dump(save_path, algo=algo)
    print(f"Model saved to {save_path}")

def load_model_and_recommend(movie_name, model_path, **kwargs):
    """
    Recommend the top 1 movie that is most similar to the movie chosen by the user
    """
    movie_name_to_id = kwargs.get('movie_name_to_id')
    movie_id_to_name = kwargs.get('movie_id_to_name')

    algo = dump.load(model_path)[1]
    movie_id = get_movie_id(movie_name.lower(), movie_name_to_id)
    if movie_id is None:
        return
    
    inner_id = algo.trainset.to_inner_iid(str(movie_id))

    similar_movie_inner_id = algo.get_neighbors(inner_id, k=1)[0]
    similar_movie_id = int(algo.trainset.to_raw_iid(similar_movie_inner_id))
    recommended_movie_name = get_movie_name(int(similar_movie_id), movie_id_to_name)

    return recommended_movie_name