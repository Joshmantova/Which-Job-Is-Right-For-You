import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from clustering import clean_descriptions
from clustering import remove_punctuation

plt.style.use('fivethirtyeight')


def get_frequent_words(tf_matrix, vectorizer, num_words):
    feature_names = vectorizer.get_feature_names()
    feature_frequencies = np.sum(tf_matrix.toarray(), axis=0)
    order = np.argsort(feature_frequencies)[::-1]
    top_n_words = []
    top_n_word_freqs = []
    for idx in order[:num_words]:
        top_n_words.append(feature_names[idx])
        top_n_word_freqs.append(feature_frequencies[idx])
    return top_n_words, top_n_word_freqs

def flatten_descriptions(descriptions):
    flattened_descriptions = ''
    for description in descriptions:
        flattened_descriptions += (description.lower() + ' ')
    return flattened_descriptions

def generate_wordcloud(flat_descriptions, width=800, height=800, background_color='white', min_font_size=10):
    return WordCloud(width=800, height=800,
                     background_color='white',
                     min_font_size=10).generate(flat_descriptions)

if __name__ == '__main__':
    # Loading in the data
    df = pd.read_csv('../Datasets/df_all_linkedin.csv')
    descriptions = df['Description'].values

    # Updating NLTK's stopwords with one's I've identified from this dataset
    stopWords = set(stopwords.words('english'))
    add_stopwords = {
        'join', 'work', 'team', 'future', 'digital', 'technology', 'access', 'leader', 'industry', 'history', 'innovation',
        'year', 'customer', 'focused', 'leading', 'business', 'ability', 'country', 'employee', 'www', 'seeking',
        'location', 'role', 'responsible', 'designing', 'code', 'ideal', 'candidate', 'also', 'duty', 'without', 'excellent',
        'set', 'area', 'well', 'use', 'strong', 'self', 'help', 'diverse', 'every', 'day', 'equal', 'employment', 'opportunity',
        'affirmative', 'action', 'employer', 'diversity', 'qualified', 'applicant', 'receive', 'consideration', 'regard',
        'race', 'color', 'religion', 'sex', 'national', 'origin', 'status', 'age', 'sexual', 'orientation', 'gender',
        'identity', 'disability', 'marital', 'family', 'medical', 'protected', 'veteran', 'reasonable', 'accomodation',
        'protect', 'status', 'equal', 'discriminate', 'inclusive'
    }
    stopWords_full = stopWords.union(add_stopwords)

    # Initializing the punctuation remover and lemmatizer
    tokenize_remove_punct = RegexpTokenizer(r'\w+')
    lemma = WordNetLemmatizer()

    # Cleaning the descriptions
    cleaned_descriptions = clean_descriptions(stopWords_full, descriptions)

    # Vectorizing cleaned descriptions and creating TF matrix
    vectorizer = CountVectorizer(stop_words='english')
    tf = vectorizer.fit_transform(cleaned_descriptions)

    # Getting the most frequent words and their frequencies after cleaning
    top_n_words, top_n_word_freqs = get_frequent_words(tf, vectorizer, 10)

    # Plotting them    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(top_n_words[::-1], top_n_word_freqs[::-1])
    ax.set_ylabel('Top 10 Words')
    ax.set_xlabel('Frequencies')
    ax.set_title('Top 10 Words and Their Frequencies')
    plt.tight_layout()
    plt.show()
    # plt.savefig('../imgs/top_10_words_and_frequencies.png')

    # Word cloud with only punctuation removed.
    no_punct_descriptions = remove_punctuation(descriptions)

    flattened_descriptions = flatten_descriptions(no_punct_descriptions)
    wordcloud = generate_wordcloud(flattened_descriptions)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # plt.savefig('../imgs/wordcloud_only_punct_removed.png')

    # Word cloud with cleaned descriptions
    flat_cleaned_descriptions = flatten_descriptions(cleaned_descriptions)

    wordcloud_cleaned = generate_wordcloud(flat_cleaned_descriptions)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud_cleaned)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # plt.savefig('../imgs/wordcloud_cleaned_descriptions.png')

    # Visualizing the data with PCA
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords_full, max_features=50000)
    tfidf = tfidf_vectorizer.fit_transform(cleaned_descriptions).toarray()

    pca = PCA(n_components=2, random_state=0)
    pca_tfidf = pca.fit_transform(tfidf)

    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(pca_tfidf[:, 0], pca_tfidf[:, 1],
            cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First two PCA dimensions")
    ax.set_xlabel("1st eigenvector (PC1)")
    ax.set_ylabel("2nd eigenvector (PC2)")
    plt.tight_layout()
    plt.show()
    # plt.savefig('../imgs/first_two_pca_dimensions.png')
