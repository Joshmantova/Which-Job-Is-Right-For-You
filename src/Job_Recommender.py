import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

from recommender import Recommender

def urlify(string):
    str_list = string.split()
    search_keywords = '%20'.join(str_list)
    url = f'https://www.linkedin.com/jobs/search?keywords={search_keyswords}&location=&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0'
    return url

if __name__ == '__main__':
    pass