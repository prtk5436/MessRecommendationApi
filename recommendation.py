import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_data():
    mess_data = pd.read_csv('menu.csv.zip')
    mess_data['original_title'] = mess_data['original_title'].str.lower()
    return mess_data


def combine_data(data):
    data_recommend = data.drop(columns=['mess_id', 'original_title', 'description'])
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)

    data_recommend = data_recommend.drop(columns=['mess_type', 'categories'])
    return data_recommend


def transform_data(data_combine, data_plot):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_plot['description'])

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim


def recommend_movies(title, data, combine, transform):
    indices = pd.Series(data.index, index=data['original_title'])
    index = indices[title]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]

    mess_indices = [i[0] for i in sim_scores]

    mess_id = data['mess_id'].iloc[mess_indices]
    mess_title = data['original_title'].iloc[mess_indices]
    mess_categories = data['categories'].iloc[mess_indices]

    recommendation_data = pd.DataFrame(columns=['Mess_Id', 'Name', 'Categories'])

    recommendation_data['Mess_Id'] = mess_id
    recommendation_data['Name'] = mess_title
    recommendation_data['Categories'] = mess_categories

    return recommendation_data


def results(mess_name):
    mess_name = mess_name.lower()

    find_mess = get_data()
    combine_result = combine_data(find_mess)
    transform_result = transform_data(combine_result, find_mess)

    if mess_name not in find_mess['original_title'].unique():
        return 'Mess not in Database'

    else:
        recommendations = recommend_movies(mess_name, find_mess, combine_result, transform_result)
        return recommendations.to_dict('records')