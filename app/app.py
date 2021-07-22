import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction import text 

df_stopWords = pd.read_csv("data/Stop_words.csv")['Word']
stopWords = text.ENGLISH_STOP_WORDS.union(df_stopWords)

CV_model = joblib.load('models/CV_model.pkl')


def k_mean_preprocess(data):
    pass


def preprocess(data):
    vector_data = CV_model.transform([data])
    return vector_data
    

def top_10_words(LDA_model, label):
    topic = LDA_model.components_[label]
    return [CV_model.get_feature_names()[i] for i in topic.argsort()[-10:]]