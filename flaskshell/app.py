from flask import Flask, render_template, request
from recommender import Recommender
app = Flask(__name__)

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/recommender', methods=['POST'])
def recommender():
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
    return f''' <h2>Recommended job titles: {recs}</h2>'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=True,)
