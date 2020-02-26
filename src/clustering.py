import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem import SnowballStemmer

import pyLDAvis
import pyLDAvis.gensim

def remove_stopwords(stopWords, descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            if word not in stopWords:
                temp_list.append(word.lower())
        cleaned_descriptions.append(' '.join(temp_list))
    return np.array(cleaned_descriptions)

def remove_punctuation(descriptions):
    no_punct_descriptions = []
    for description in descriptions:
        description_no_punct = ' '.join(RegexpTokenizer(r'\w+').tokenize(description))
        no_punct_descriptions.append(description_no_punct)
    return np.array(no_punct_descriptions)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
               'N': wordnet.NOUN,
               'V': wordnet.VERB,
               'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_descriptions(descriptions):
    cleaned_descriptions = []
    for description in descriptions:
        temp_list = []
        for word in description.split():
            cleaned_word = WordNetLemmatizer().lemmatize(word, get_wordnet_pos(word))
            temp_list.append(cleaned_word)
        cleaned_descriptions.append(' '.join(temp_list))
    return np.array(cleaned_descriptions)

def clean_descriptions(stopWords, descriptions):
    no_punct = remove_punctuation(descriptions)
    no_punct_sw = remove_stopwords(stopWords, no_punct)
    cleaned = lemmatize_descriptions(no_punct_sw)
    return cleaned

def get_representative_words(vectorizer, kmeans):
    sorted_centroids = []
    for cluster in kmeans.cluster_centers_:
        top_10 = np.argsort(cluster)[::-1]
        sorted_centroids.append(top_10[:10])
    for idx, c in enumerate(sorted_centroids):
        print(f'\nCluster {idx}\n')
        for idx in c:
            print(vectorizer.get_feature_names()[idx])

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print('Topic %d:' % (topic_idx))
        print(' '.join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, get_wordnet_pos(text)))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stopWords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

if __name__ == '__main__':

    # Reading in data
    df = pd.read_csv('../Datasets/df_all_linkedin.csv', index_col=0)
    df_co = pd.read_csv('../Datasets/df_linkedin_Colorado.csv', index_col=0)

    descriptions = df['Description'].values
    descriptions_co = df_co['Description'].values

    # Creating stop words
    stopWords = set(stopwords.words('english'))
    add_stopwords = {
        'join', 'work', 'team', 'future', 'digital', 'technology', 'access', 'leader', 'industry', 'history', 'innovation',
        'year', 'customer', 'focused', 'leading', 'business', 'ability', 'country', 'employee', 'www', 'seeking',
        'location', 'role', 'responsible', 'designing', 'code', 'ideal', 'candidate', 'also', 'duty', 'without', 'excellent',
        'set', 'area', 'well', 'use', 'strong', 'self', 'help', 'diverse', 'every', 'day', 'equal', 'employment', 'opportunity',
        'affirmative', 'action', 'employer', 'diversity', 'qualified', 'applicant', 'receive', 'consideration', 'regard',
        'race', 'color', 'religion', 'sex', 'national', 'origin', 'status', 'age', 'sexual', 'orientation', 'gender',
        'identity', 'disability', 'marital', 'family', 'medical', 'protected', 'veteran', 'reasonable', 'accomodation',
        'protect', 'status', 'equal', 'discriminate', 'inclusive', 'diverse'
    }
    stopWords = stopWords.union(add_stopwords)

    # Initializing punctuation remover and lemmatizer
    tokenize_remove_punct = RegexpTokenizer(r'\w+')
    lemma = WordNetLemmatizer()

    # Cleaning descriptions for both the whole dataset and CO only
    cleaned_descriptions = clean_descriptions(stopWords, descriptions)

    # descriptions_no_sw_co = remove_stopwords(stopWords, descriptions_co)
    # descriptions_no_sw_punct_co = remove_punctuation(tokenize_remove_punct, descriptions_no_sw_co)
    # cleaned_descriptions_co = lemmatize_descriptions(lemma, descriptions_no_sw_punct_co)

    # Vectorizing words creating both tf and tf-idf matrices
    vectorizer = CountVectorizer(stop_words=stopWords, min_df=.15, max_df=0.75, max_features=5000)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords, min_df=.15, max_df=0.75, max_features=5000)
    tfidf = tfidf_vectorizer.fit_transform(cleaned_descriptions).toarray()
    tf = vectorizer.fit_transform(cleaned_descriptions)

    # Initializing and fitting k-means model
    kmeans = KMeans(n_clusters=5, verbose=True, n_jobs=-1)
    kmeans.fit(tfidf)

    # Returning most representative words for each cluster
    get_representative_words(tfidf_vectorizer, kmeans)

    # Calculating model score for kmeans
    silhouette_score(tfidf, kmeans.labels_)
    kmeans.score(tfidf)

    # Initializing and running LDA model
    feature_names = vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=4, 
                                    max_iter=10, learning_method='online', 
                                    random_state=0, verbose=True, n_jobs=-1)

    lda.fit(tf)

    # Displaying most representative words for each cluster of LDA
    num_top_words=10
    display_topics(lda, feature_names, num_top_words)

    # LDA in gensim
    # Processing text with gensim
    data_text = df[['Description']].copy()
    data_text['index'] = data_text.index
    documents = data_text

    stemmer = SnowballStemmer('english')
    processed_docs = documents['Description'].map(preprocess)

    # Vectorizing text in gensim
    id2word = gensim.corpora.Dictionary(processed_docs)
    id2word.filter_extremes(no_below=80, no_above=.75, keep_n=5000)
    texts = processed_docs
    bow_corpus = [id2word.doc2bow(text) for text in texts]

    # LDA model
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=id2word, passes=10, random_state=0)

    # Visualizing LDA with PyLDAvis
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, id2word)
    pyLDAvis.show(vis)