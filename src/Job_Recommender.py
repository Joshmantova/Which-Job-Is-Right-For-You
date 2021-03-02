import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

from recommender import Recommender

def urlify(string):
    str_list = string.split()
    search_keywords = '%20'.join(str_list)
    url = f'https://www.linkedin.com/jobs/search?keywords={search_keywords}&location=&trk=guest_job_search_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0'
    return url

if __name__ == '__main__':
    st.title('How it works')
    instructions = """
    Below, you will see several checkboxes that correspond to different Data Science related skills.
    Please select the skills that you feel comfortable with and based on which skills you have and don't have,
    a job will be recommended to you. This recommendation process utilizes both the skills you have and the skills
    you don't have. Happy job searching!
    """
    st.write(instructions)
    list_of_skills = [
        "Python", "R", "Spark", "SPSS", "SQL", "Pandas", "Numpy", "Cloud", "Docker",
        "Statistic", "Java", "Scala", "Marketing", "SAS", "Stata", "Excel", "Tableau"
        ]
    skills_dict = dict()
    for skill in list_of_skills:
        skills_dict[skill] = st.checkbox(skill)
    
    user_vector = list(skills_dict.values())
    done = st.button('Recommend Me A Job!')
    if done:
        r = Recommender(user_vector)
        recs = r.recommend()
        descrip = r.rec_descrip
        str_recs = ' '.join(recs)
        st.write(f"Recommended job title: {recs[0]}")
        st.write(f"Company: {recs[1]}")
        st.write(f"Location: {recs[2]}")
        st.write(f"Job Description: {descrip}")
        link = f"[Try To Find This Job On Linkedin]({urlify(str_recs)})"
        st.write(link)