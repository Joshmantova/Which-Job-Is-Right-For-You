from flask import Flask, render_template, request
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

application = Flask(__name__)

def urlify(string):
    '''Takes string of Linkedin search query and returns url for that query'''
    str_list = string.split()
    search_keywords = '%20'.join(str_list)
    url = f'https://www.linkedin.com/jobs/search?keywords={search_keywords}&location=&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0'
    return url

@application.route('/')
def landing_page():
    '''Asks user for their profile of skills'''
    return render_template('index.html')

@application.route('/recommender', methods=['POST'])
def recommender():
    '''Creates user profile and recommends job'''
    skill1 = request.form.get('skill1') != None
    skill2 = request.form.get('skill2') != None
    skill3 = request.form.get('skill3') != None
    skill4 = request.form.get('skill4') != None
    skill5 = request.form.get('skill5') != None
    skill6 = request.form.get('skill6') != None
    skill7 = request.form.get('skill7') != None
    skill8 = request.form.get('skill8') != None
    skill9 = request.form.get('skill9') != None
    skill10 = request.form.get('skill10') != None
    skill11 = request.form.get('skill11') != None
    skill12 = request.form.get('skill12') != None
    skill13 = request.form.get('skill13') != None
    skill14 = request.form.get('skill14') != None
    skill15 = request.form.get('skill15') != None
    skill16 = request.form.get('skill16') != None
    skill17 = request.form.get('skill17') != None
    user_vector = [skill1, skill2, skill3, skill4, skill5, 
                skill6, skill7, skill8, skill9, skill10, skill11, 
                skill12, skill13, skill14, skill15, skill16, skill17]
    r = Recommender(user_vector)
    recs = r.recommend()
    descrip = r.rec_descrip
    str_recs = ' '.join(recs)
    return render_template('base.html') + f''' 
                <h4>Recommended job title: {recs[0]} <br><br>
                Company: {recs[1]}<br><br>
                Location: {recs[2]}<br><br>
                Job Description: {descrip}</h4>
                <br><br>
                <a href='{urlify(str_recs)}'> Go to Linkedin job posting</a>
                <br><br>
            '''

@application.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    application.run(host="0.0.0.0", port=80)
    # app.run(host='0.0.0.0', port=8000, threaded=True, debug=True,)
