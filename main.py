from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
import time
import random
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
# from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import SVD
from surprise import SlopeOne
from surprise import Dataset
import random

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Utils ===============


def map_genre(genre):
    return ""+genre+"==1"


# =======================DATA=========================
data = pd.read_csv("./dataset/ml-100k/movie_info.csv")
movies_cnt = len(set(data['movie_id']))
fisrt_display_num = 10
first_rec_num = 12
second_rec_num = 5
global uid
algo_type = "KNNBasic"
# get uid set

# ========== Frontend ==========
global movie_details
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('dataset/ml-100k/u.data', sep='\t',
                      names=r_cols, encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date',
    'video_release_date', 'imdb_url']
movies = pd.read_csv('dataset/ml-100k/u.item', sep='|',
                     names=m_cols, usecols=range(5), encoding='latin-1')

movie_ratings = pd.merge(movies, ratings)
movie_details = pd.merge(data, movie_ratings)

# =================== Body =============================


class Movie(BaseModel):
    movie_id: int
    movie_title: str = 'default'
    release_date: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =
# uid random assign
@app.get("/api/index_assign")
def get_user_id():
    global uid
    if os.path.exists("./new_u.data"):
        ori_data = pd.read_csv(
            "./new_u.data", delimiter='\t', names=['user', 'item', 'rating', 'timestamp'])
    else:
        ori_data = pd.read_csv("./dataset/ml-100k/u.data", delimiter='\t',
                               names=['user', 'item', 'rating', 'timestamp'])
    ### self add uid mode ###
    # if user rating, new_u.data will record exists uids
    # take max uid in new_u.data and add 1
    print(ori_data['user'].values.max())
    uid = ori_data['user'].values.max() + 1
    print('assign uid ', uid)
    return int(uid)

# show four genres


@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}

# choose algo type


@app.get("/api/algolib_get")
def get_algo():
    return {'algo type': ["SVD", "KNNBasic", "SlopeOne"]}

# algo_type assign


@app.get("/api/algo_selection/{post_type}")
def post_algo(post_type: str):
    global algo_type
    print(post_type)
    algo_type = post_type
    return None

# get genres moive


@app.post("/api/movies")
def get_movies(genre: list):
    print('select genre: ', genre)
    query_str = " or ".join(map(map_genre, genre))
    results = data.query(query_str)
    results.loc[:, 'score'] = None
    results = results.sample(fisrt_display_num).loc[:, [
                             'movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))

# 1st recommend


@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
    user_add(movies)
    res = get_initial_items(n=first_rec_num)
    res = [int(i) for i in res]
    if len(res) > first_rec_num:
        res = res[:first_rec_num]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, [
        'movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))

# 2nd recommend and update recommend list


@app.get("/api/add_recommend")
async def add_recommend(item_id: int , in_score: int):
    print('add recommend')
    if algo_type == 'KNNBasic':
        res = get_similar_items(str(item_id), n=second_rec_num)
    elif algo_type == 'SVD' or algo_type == 'SlopeOne':
        movie = Movie(movie_id=int(item_id), score=in_score,
                      release_date=str(int(time.time())))
        # add []
        user_add([movie])
        res = get_initial_items(n=second_rec_num)

    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, [
        'movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


def user_add(movies):
    global uid
    if not os.path.exists('./new_u.data'):
        print('new_u not exists, add new data to u.data')
        df = pd.read_csv('./dataset/ml-100k/u.data', delimiter='\t')
        df.to_csv('new_' + 'u.data', index=False, sep='\t')
    else:
        df = pd.read_csv('./new_u.data')

    with open(r'new_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        for movie in movies:
            s = [str(uid), str(movie.movie_id), int(
                movie.score), str(int(time.time()))]
            data_input.append(s)

        for k in data_input:
            wf.writerow(k)

    with open('new_u.data') as f:
        print('new_data_len', len(f.readlines()))


def get_train_data(file='new_u.data'):
    file_path = os.path.expanduser(file)
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()

    return trainset


def get_initial_items(n=12):
    res = []
    trainset = get_train_data()

    if algo_type == 'KNNBasic':
        algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    elif algo_type == 'SVD':
        algo = SVD()
    elif algo_type == 'SlopeOne':
        algo = SlopeOne()

    algo.fit(trainset)
    dump.dump('./model_' + algo_type, algo=algo, verbose=1)
    all_results = {}
    for i in range(movies_cnt):
        iid = str(i)
        pred = algo.predict(uid, iid).est
        all_results[iid] = pred

    sorted_list = sorted(all_results.items(),
                         key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res


def get_similar_items(iid, n=12):
    algo = dump.load('./model_' + algo_type)[1]
    inner_id = algo.trainset.to_inner_iid(iid)
    print(inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid

# ========== Frontend API ==========


@app.get("/api/top_movies")
def get_top_movies():
    global movie_details
    movie_stats = movie_details.groupby(
        'movie_id').agg({'rating': [np.size, np.mean]})
    atleast_100 = movie_stats['rating']['size'] >= 100
    top10 = movie_stats[atleast_100].sort_values(
        [('rating', 'mean')], ascending=False)[:10]
    return top10.index.values.tolist()


@app.get("/api/movie_detail")
def get_movie_detail(item_id):
    global movie_details
    item_data = movie_details[movie_details['movie_id'] == int(item_id)]
    genres_table = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                       "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                       "Romance", "Sci_Fi", "Thriller", "War", "Western"]
    genre_rt = []
    for genre in genres_table:
        if item_data[genre].iloc[0] > 0:
            genre_rt.append(genre)
    rt = {'title':item_data['title'].iloc[0], 
          'size':len(item_data.index), 
          'mean':round(item_data["rating"].mean(),1),
          'poster':item_data['poster_url'].iloc[0],
          'genres':genre_rt}
    return rt