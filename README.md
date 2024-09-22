# Streamlit Movie Recommender

This project is a simple movie recommendation system built using the `surprise` library for collaborative filtering and `Streamlit` for the web interface. The recommender system is trained on the MovieLens 100k dataset (https://grouplens.org/datasets/movielens/100k/) and allows users to select a movie to receive similar movie recommendations. 

Note: the movie that the user selects have to be released before 1998. e.g. Toy Story (1995)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dockerization](#dockerization)
- [Known Issues](#known-issues)

## Installation

If you do not use a docker to run this app, you can manually set up the environment by first install the dependencies in `requirements.txt`. But it is recommended to use the dokerized version to run the app. Pleaw refer to [Dockerization] for more information. 
```bash
pip install -r requirements.txt
```

### Prerequisites

- Python 3.12
- Docker (optional, for containerization)

### Clone the Repository

```bash
git clone https://gitlab.code.hfactory.io/xianghan.mei/movie-recommender.git
cd movie-recommender
```

### Install the dependencies with requirements.txt

You can install the required Python packages using `pip`. It is not recommended to manually `pip install` all the packages required. It is recommended to follow the instruction in the [dockerization] step. 
```bash
pip install -r requirements.txt
```

## Usage
To run the application locally
```bash
streamlit run movie_recommender.py
```
This command will start a local server, and you can access the application in your web browser at `http://localhost:8501` or `http://0.0.0.0:8501`.

## Project Structure

```plaintext
movie-recommender/
│
├── model/
│   └── knn_movie_recommender.model  # Trained model file
│
├── utils/
│   ├── __init__.py
│   ├── utils.py                     
│
├── ml-100k/                         
│
├── movie_recommender.py             
├── train.py                         
├── Dockerfile                       
├── requirements.txt                 
└── README.md                        
```
* `model` saves the trained models. 
* `ml-100k` saves the data downloaded from https://grouplens.org/datasets/movielens/100k/. You can also download other data from movielens that have the same format and replace the ml-100k dataset to train a new model for recommendation. 
* `train.py` contains the code to train and save a new recommender
* `movie_recommender.py` is the main script for this project, where you receive input from user and use the trained model to predict a movie the user would like. 

## Dockerization

You can containerize the application using Docker. Follow the steps below:

1. Build the Docker Image
```bash
docker build -t streamlit-movie-recommender .
```
2. Run the Docker Container
```bash
docker run -p 8501:8501 streamlit-movie-recommender
```
3. Access the application at `http://localhost:8501` in your web browser.


## Known Issues
* `ImportError: numpy.core.multiarray failed to import`: This issue occurs due to a compatibility problem between `numpy` and `scikit-surprise`. Ensure that you are using a compatible version of `numpy` (e.g., 1.26.4).

* Port Conflicts: If port `8501` is already in use, either free up the port or run the container on a different port by modifying the Docker run command.

