import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

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

def get_representative_jobs(df, kmeans):
    cluster_centers = kmeans.cluster_centers_
    for cent in cluster_centers:
        print('\nCluster Represnetations')
        dist = euclidean_distances(cent.reshape(1,-1), tfidf)
        order = np.argsort(dist)
        for o in order[0][:5]:
            title = df['Job_Title'].iloc[o]
            print(title)

if __name__ == '__main__':

    # Reading in data
    df = pd.read_csv('../Datasets/df_all_linkedin.csv', index_col=0)

    descriptions = df['Description'].values

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

    # Vectorizing words creating both tf and tf-idf matrices
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords, min_df=.15, max_df=0.75, max_features=5000)
    tfidf = tfidf_vectorizer.fit_transform(cleaned_descriptions).toarray()

    # Initializing and fitting k-means model
    kmeans = KMeans(n_clusters=3, n_jobs=-1)
    kmeans.fit(tfidf)

    # Returning most representative words for each cluster
    get_representative_jobs(df, kmeans)

    # Calculating model score for kmeans
    silhouette_score(tfidf, kmeans.labels_)
    kmeans.score(tfidf)

    # Creating elbow plot for selecting k value. Takes forever to run so it's commented out

    # k_values = [i for i in range(2, 1000)]
    # ss_scores = []
    # for k in k_values:
    #     kmeans = KMeans(n_clusters=k, n_jobs=-1, random_state=0)
    #     kmeans.fit(tfidf)
    #     ss = silhouette_score(tfidf, kmeans.labels_)
    #     ss_scores.append(ss)
    
    # fig, ax = plt.subplots()
    # ax.plot(k_values, ss_scores)
    # ax.set_xlabel('K Values')
    # ax.set_ylabel('Silhouette Score')
    # ax.set_title('Selection of K-Value')
    # plt.tight_layout()
    # plt.savefig('../imgs/selection_kvalue.png')
    
    #Visualizing k-means clusters with PCA graph
    kmeans_model = kmeans
    labels=kmeans_model.labels_.tolist()

    pca = PCA(n_components=2).fit(tfidf)
    datapoint = pca.transform(tfidf)

    plt.figure

    label1 = ["#FFFF00", "#008000", "#0000FF"]
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:,0], centroidpoint[:,1], marker='^', s=150, c="#000000", label='Cluster Centers')
    plt.xlabel('First PCA Dimension')
    plt.ylabel('Second PCA Dimension')
    plt.title('K-Means Clusters')
    plt.legend(fontsize='x-small')
    plt.text(0.44,0.6, 'Blue: Mobile devs', fontsize=9)
    plt.text(0.44, 0.5, 'Yellow: Data science', fontsize=9)
    plt.text(0.44, 0.4, 'Green: Big data dev', fontsize=9)
    plt.tight_layout()
    plt.savefig('../imgs/pca_kmeans_3_clusters.png');