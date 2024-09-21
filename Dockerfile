FROM python:3.12
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libatlas-base-dev \
    --no-install-recommends
RUN pip install --no-cache-dir cython
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir streamlit scikit-surprise
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "movie_recommender.py", "--server.port=8501", "--server.address=0.0.0.0"]
