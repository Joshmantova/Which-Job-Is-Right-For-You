import pandas as pd
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
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

df = pd.read_csv('../Datasets/df_all_linkedin.csv', index_col=0)
df.drop_duplicates(subset='Description', keep='first', inplace=True)
descriptions = df['Description'].values
job_titles = df['Job_Title'].values

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

tokenize_remove_punct = RegexpTokenizer(r'\w+')
lemma = WordNetLemmatizer()

descriptions = clean_descriptions(stopWords, descriptions)

tokenized_descriptions = []
tokenized_descriptions_tagged = []
for idx, descrip in enumerate(descriptions):
    tok_descrip = gensim.utils.simple_preprocess(descrip)
    corpus = gensim.models.doc2vec.TaggedDocument(tok_descrip, [idx])
    tokenized_descriptions_tagged.append(corpus)
    tokenized_descriptions.append(tok_descrip)

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, dm=1)
model.build_vocab(tokenized_descriptions_tagged)
model.train(tokenized_descriptions_tagged, total_examples=model.corpus_count, epochs=model.epochs)

vectorized_docs = []
for descrip in tokenized_descriptions:
    inferred_vec = model.infer_vector(descrip)
    vectorized_docs.append(inferred_vec)

kmeans = KMeans(n_clusters=7, n_jobs=-1)
kmeans.fit(vectorized_docs)
cluster_centers = kmeans.cluster_centers_

for idx, cluster in enumerate(cluster_centers):
    print(f'\n Cluster #{idx}: ')
    distances = euclidean_distances(cluster.reshape(1,-1), vectorized_docs)
    order = np.argsort(distances)[::-1].flatten()
    for o in order[:11]:
        print(job_titles[o])
