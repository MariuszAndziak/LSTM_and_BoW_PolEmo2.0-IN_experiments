from html import unescape
import re
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import catboost as ctb
import lightgbm as lgb
import mlflow
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import datetime
import spacy
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from scikitplot.metrics import plot_confusion_matrix

def re_urls(text, replace_for="URL"):
    return re.sub(r'https?://\S+', replace_for, text) 

def re_user_mentioned(text, replace_for=r'\1'):    
    return re.sub(r'@(\S+)', replace_for, text)

def re_digits(text, replace_for='DIGIT'): 
    result = re.sub(r'\d+', replace_for, text) 
    return result

def re_multi_spaces_into_one(text, replace_for=' '):
    return re.sub(r'\s+', replace_for, text)

def re_topic(text, replace_for=r'\1'):
    return re.sub(r'#(\S+)', replace_for, text)
    
    
def make_unescape(text):
    return unescape(text)


def preprocessing(doc):
    doc = make_unescape(doc)
    doc = re_urls(doc)
    doc = re_user_mentioned(doc)
    doc = re_digits(doc)
    doc = re_multi_spaces_into_one(doc)
    doc = re_topic(doc)
    
    return doc

def simple_tokens(sentence):
    one_ws_sent = " ".join(sentence.split())
    return [x for x in one_ws_sent.split(' ')]


def get_models(use_dummy, use_specific_models= None):
    
    models = [('dummy', DummyClassifier(strategy='stratified'))] if use_dummy else []
    
    models += [
        ('decision tree', DecisionTreeClassifier(max_depth=5)),
        ('random forest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ('extra-trees', ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ('lightgbm', lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ('catboost', ctb.CatBoostClassifier(n_estimators=50, max_depth=5, random_state=0, verbose=0)),
        ('xgboost', xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=0)),
    ]
    

    if use_specific_models !=None:
        models = [m for m in models if m[0] in use_specific_models]
    if use_dummy:
        models +=[('dummy', DummyClassifier(strategy='stratified'))] 
    
    return models


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_ = true_positives / (predicted_positives + K.epsilon())
    return precision_


def f1(y_true, y_pred):    
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


def get_or_create_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name)
        return mlflow.get_experiment_by_name(name)    
    return experiment


def _eid(name):
    return get_or_create_experiment(name).experiment_id

def get_y(df, text_col_name):
    y = df[text_col_name].factorize()[0]
    if y.ndim == 1:
        y = to_categorical(y)      
    return y

def random_polemo2_opinion(sentiment, df):
    if sentiment == 'poz':
        sent = '__label__meta_plus_m'
    elif sentiment == 'neg':
        sent = '__label__meta_minus_m'
    elif sentiment == 'zero':
        sent = '__label__meta_zero'
    elif sentiment == 'amb':
        sent = '__label__meta_amb'
        
    nd = df[df['target'] == sent]
    return nd.sample().head(1)['sentence'].values[0]

nlp_sm = spacy.load('pl_core_news_sm')
nlp_md = spacy.load('pl_core_news_md')

spacy_stop_words_sm = nlp_sm.Defaults.stop_words
spacy_stop_words_md = nlp_md.Defaults.stop_words

def get_stopwords(sw):
    all_stopwords = set()
    all_stopwords |= sw
    
    return all_stopwords

def polish_tokenizer_sm(doc):
    return [token.lemma_ for token in nlp_sm(doc) if not token.is_punct and token.lemma_ != "-PRON-"]

def polish_tokenizer_md(doc):
    return [token.lemma_ for token in nlp_md(doc) if not token.is_punct and token.lemma_ != "-PRON-"]


def run_models(X, y, scoring, cv=3, plot_result=True, show_confusion_matrix=True, use_dummy=True):
    result = []
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    
    num_cols = 4
    if show_confusion_matrix:
        fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(15,10))


    for it, (model_name, model) in enumerate(get_models(use_dummy)):
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        mean = np.around( np.mean(scores), 2)
        std = np.around( np.std(scores), 2)

        print("model={}, {}: mean={}, std={}".format(model_name, scoring, mean, std))
        
        result.append((model_name, mean, std))
        
    
    
        if show_confusion_matrix:
            y_pred = cross_val_predict(model, X, y, cv=cv)
            ax = axes[it // num_cols, it % num_cols]
            plot_confusion_matrix(y, y_pred, ax=ax, title='model: {}'.format(model_name))
            
    if show_confusion_matrix:
        plt.tight_layout()  
        plt.show()

        

    return result


def calculate_delta(td):
    days = td.days
    seconds = td.seconds
    hours = seconds//3600
    minutes = (seconds//60)%60
    sec = (seconds%60)
    print(f'''Experiment took:
    {days} days
    {hours} hours
    {minutes} minutes
    {sec} seconds''')