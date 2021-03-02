FROM python:3.8

COPY . /app

WORKDIR /app/src

RUN pip --no-cache-dir install -r ../requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "Job_Recommender.py" ]