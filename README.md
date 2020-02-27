# Introduction:

* Goal of project is to match people up with jobs

* Jobs were scraped from Linkedin using Selenium and Beautiful Soup

# Vectorizing jobs

* Jobs were assessed on several skills

python, r, spark, spss, sql, pandas, numpy, cloud, docker, statistic, java, scala, marketing, sas, stata, excel, tableau

* Vector for each job was created to represent whether or not the job included each of the skills
    * Ex: [1, 1, 1, 1, 0, 1, 1, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 1]

* User vectors were created by asking users which of the skills they had experience with

* User vector was compared to all job vectors using Jaccard Similarity
    * Most similar job was returned
