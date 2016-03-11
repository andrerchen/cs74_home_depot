# Based off of public Kaggle scripts:
# 1. https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
# 2. https://www.kaggle.com/the1owl/home-depot-product-search-relevance/rfr-features-0-47326

import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer

# Rebuild data frame?
REBUILD = False
GRID_SEARCH_RF = True
BASIC_RF = False
USE_SIMS = True

# FUNCTIONS FOR RF WITH GRID SEARCH
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


stemmer = SnowballStemmer('english')

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1") # UPDATE THIS
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")   # UPDATE THIS
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('data/product_descriptions.csv')      # UPDATE THIS 

# USE OF COSINE SIMILARITIES (from gensim) IS OPTIONAL
if USE_SIMS:
    df_sims_train = pd.read_csv('data/train_run_v1.csv')            # UPDATE THIS
    df_sims_test = pd.read_csv('data/test_run_v1.csv')              # UPDATE THIS
    df_sims_train = df_sims_train.drop(['product_uid', 'relevance'], axis=1)
    df_sims_test = df_sims_test.drop('product_uid', axis=1)
    df_sims = pd.concat((df_sims_train, df_sims_test), axis=0, ignore_index=True)

print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

num_train = df_train.shape[0]

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

########## COMPILE DF_ALL ##########
if REBUILD: 
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

    # once again, optional
    if USE_SIMS:
        df_all = pd.merge(df_all, df_sims, how='left', on='id')

    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

    df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

    df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
    df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))

    # save the data frame
    df_all.to_csv('df_all.csv', encoding="utf-8")
    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))

    df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    id_test = df_test['id']

    y_train = df_train['relevance'].values
    X_train = df_train.drop(['id','relevance'],axis=1).values
    X_test = df_test.drop(['id','relevance'],axis=1).values

########## RF WITH GRID SEARCH ##########
if GRID_SEARCH_RF:
    df_all = pd.read_csv('df_all.csv', encoding="utf-8", index_col=0)

    df_all = df_all.drop(['similarity_search_term_to_brand', 'similarity_search_term_to_materials'],axis=1)

    print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

    # ACTUAL RUN
    # df_train = df_all.iloc[:num_train]
    # df_test = df_all.iloc[num_train:]
    # id_test = df_test['id']
    # y_train = df_train['relevance'].values
    # X_train =df_train[:]
    # X_test = df_test[:]

    # TEST RUN
    n_train = 50000

    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[n_train:num_train]
    y_train = df_train['relevance'].values
    y_actual = df_test['relevance'].values
    X_train = df_train[:]
    X_test = df_test[:]

    # n_estimators = 500
    rfr = RandomForestRegressor(n_estimators = 500, n_jobs = 1, random_state = 2016, verbose = 1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    # n_components = 10
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_vals()),  
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)]))
                            #('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                            ],
                        transformer_weights = {
                            'cst': 1.0,
                            'txt1': 0.5,
                            'txt2': 0.25,
                            'txt3': 0.0,
                            #'txt4': 0.5
                            },
                    n_jobs = 1
                    )), 
            ('rfr', rfr)])
    
    param_grid = {'rfr__max_features': [10, 20, 30], 'rfr__max_depth': [10, 20, 30]}
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)
    print(model.best_score_ + 0.47927)

    y_pred = model.predict(X_test)
    
    mse = fmean_squared_error(y_actual, y_pred)

    print(mse)

    #pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)


########## ORIGINAL CODE ##########
elif BASIC_RF is True:
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_sklearn_rf_v2.csv',index=False)

print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))