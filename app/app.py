import pandas as pd
import numpy as np
import joblib
import textstat
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction import text 

df_stopWords = pd.read_csv("data/Stop_words.csv")['Word']
stopWords = text.ENGLISH_STOP_WORDS.union(df_stopWords)

CV_model = joblib.load('models/CV_model.pkl')

def get_tags(text: str) -> dict:
    tokens = word_tokenize(text)
    sentence_tagged = nltk.pos_tag(tokens, tagset='universal')
    tag_fd = nltk.FreqDist(tag for (word, tag) in sentence_tagged)
    tags = dict(tag_fd)
    return tags

def number_of_words(tags: dict) -> int:
    sum = np.sum(list(tags.values()))
    if '.' in tags:
        sum -= tags['.']
    return sum

def readability(text: str) -> float:
    return textstat.flesch_reading_ease(text)

def noun_verb_ratio(tags: dict):
    sum = np.sum(list(tags.values()))
    if 'NOUN' in tags:
        noun_ratio = tags['NOUN'] / sum
    else:
        noun_ratio = 0
    if 'VERB' in tags:
        verb_ratio = tags['VERB'] / sum
    else:
        verb_ratio = 0
    
    return noun_ratio, verb_ratio

def score(noun_ratio, verb_ratio):
    targets = {'noun': 0.3, 'verb': 0.2}
    factors = {'noun': 1/(1 - targets['noun']), 'verb':1/(1 - targets['verb'])}

    noun_score = 1-abs(noun_ratio-targets['noun']) * factors['noun']
    verb_score = 1-abs(verb_ratio-targets['verb']) * factors['verb']

    return noun_score, verb_score

def mean_score(noun_score, verb_score, r_ratio):
    return (noun_score + verb_score + r_ratio) / 3

def total_score(mean_score, amount_of_words):
    if amount_of_words < 20:
        return mean_score * (100 - (20 - amount_of_words) * 5) / 100        
    return mean_score


def calculate_score(text: str) -> float:
    punctuation= '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in punctuation:
        text = text.replace(char, '')
    tags = get_tags(text)
    n_words = number_of_words(tags)

    r = readability(text)
    r_ratio = r / 100
    
    noun_ratio, verb_ratio = noun_verb_ratio(tags)
    noun_score, verb_score = score(noun_ratio, verb_ratio)
    
    m_score = mean_score(noun_score, verb_score, r_ratio)
    final_score = total_score(m_score, n_words)
    return {'final_score': final_score, 'noun_score': noun_score, 'verb_score': verb_score, 'readability': r_ratio}


def preprocess(data):
    vector_data = CV_model.transform([data])
    return vector_data
    

def top_10_words(LDA_model, label):
    topic = LDA_model.components_[label]
    return [CV_model.get_feature_names()[i] for i in topic.argsort()[-10:]]