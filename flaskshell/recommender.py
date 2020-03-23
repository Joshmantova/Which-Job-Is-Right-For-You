import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import jaccard_score

class Recommender:

    def __init__(self, datapoint):
        '''Recommender class that takes in user profile as a datapoint and recommends job to user
        :param: datapoint - array-like series of 17 features of user profile'''

        self.datapoint = datapoint
        self.skills = ['python', 'r', 'spark', 'spss', 'sql', 'pandas', 'numpy',
                        'cloud', 'docker', 'statistic', 'java', 'scala', 'marketing',
                        'sas', 'stata', 'excel', 'tableau']
        self.empty_skills_dict = self.set_dictionary()
        self.df = pd.read_csv('df_all_linkedin.csv')
        self.descriptions = self.df['Description'].values
        self.vectorized_descriptions = self.vectorize_descriptions()
    
    def set_dictionary(self):
        '''Initializes ordered dictionary with skills as keys and all values to 0'''
        skills_dict = OrderedDict()
        for skill in self.skills:
            skills_dict[skill] = 0
        return skills_dict

    def vectorize_descriptions(self):
        '''Represents descriptions as skills that description includes. Returns ordered dictionary
        with keys as skills and values as 1 if the skill is mentioned in the description'''
        vectorized_descriptions_dict = []
        for descrip in self.descriptions:
            skills_dict = self.empty_skills_dict.copy()
            for word in descrip.split():
                if word.lower() in self.skills:
                    skills_dict[word.lower()] = 1
            vectorized_descriptions_dict.append(skills_dict)
        vectorized_descriptions = []
        for vec in vectorized_descriptions_dict:
            vectorized_descriptions.append(vec.values())
        return vectorized_descriptions

    def recommend(self):
        '''Recommends job to user based on their profile'''
        js_list = []
        for descrip in self.vectorized_descriptions:
            js = jaccard_score(list(descrip), self.datapoint)
            js_list.append(js)
        order = np.argsort(js_list)[::-1]
        o = order[:1]
        rec = (self.df['Job_Title'].iloc[o].values[0], self.df['Company'].iloc[o].values[0], self.df['Location'].iloc[o].values[0])
        rec_descrip = self.df['Description'].iloc[o].values[0]
        self.rec_descrip = rec_descrip
        self.rec = rec
        return rec

if __name__ == '__main__':
    '''Skills: ['python', 'r', 'spark', 'spss', 'sql', 'pandas', 'numpy',
                        'cloud', 'docker', 'statistic', 'java', 'scala', 'marketing',
                        'sas', 'stata', 'excel', 'tableau']'''
    example_datapoint = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    recommender = Recommender(example_datapoint)
    rec = recommender.recommend()
    print(recommender.rec_descrip)
