import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import logging
from logging.handlers import RotatingFileHandler
# import logging
app = Flask(__name__)
handler = RotatingFileHandler('flask.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
# logging.basicConfig(level=logging.DEBUG) 
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    # app.logger.info('This is an info message')
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if all(key in request.form for key in ('job_profile_name_analysis', 'attained_questions_analysis', 'score_analysis')):
        job_profile_name= request.form['job_profile_name_analysis']
        attained_questions_analysis = float(request.form['attained_questions_analysis'])
        if(attained_questions_analysis==0):
            attained_questions_analysis=1
        score_analysis = float(request.form['score_analysis'])
        correctness = round((score_analysis * 100) / attained_questions_analysis, 2)
        if(job_profile_name==' Cybersecurity Analyst'):
            job_profile_name_analysis=0
        elif(job_profile_name=='a'):
            job_profile_name_analysis=1
        elif(job_profile_name=='Android Developer'):
            job_profile_name_analysis=2
        elif(job_profile_name=='Artificial Intelligence (AI) Engineer'):
            job_profile_name_analysis=3
        elif(job_profile_name=='Cloud Architect'):
            job_profile_name_analysis=4
        elif(job_profile_name=='Data Analyst'):
            job_profile_name_analysis=5
        elif(job_profile_name=='Database Administrator'):
            job_profile_name_analysis=6
        elif(job_profile_name=='DevOps Engineer'):
            job_profile_name_analysis=7
        elif(job_profile_name=='Full Stack Developer'):
            job_profile_name_analysis=8
        elif(job_profile_name=='IoT Specialist'):
            job_profile_name_analysis=9
        elif(job_profile_name=='Software Engineer'):
            job_profile_name_analysis=10
        
        data_to_predict=[attained_questions_analysis,score_analysis,correctness,job_profile_name_analysis]
        int_features = [float(x) for x in data_to_predict]
        # print("Values received from form:", int_features)
        final_features = [np.array(int_features)]
        predicted_performance = model.predict(final_features)

        if(predicted_performance[0]==0):
            output=' Needs Improvement'
        elif(predicted_performance[0]==1):
            output=' Extremely Poor'
        elif(predicted_performance[0]==2):
            output=' Very Poor'
        elif(predicted_performance[0]==3):
            output='Poor'
        elif(predicted_performance[0]==4):
            output=' Below Average'
        elif(predicted_performance[0]==5):
            output=' Average'
        elif(predicted_performance[0]==6):
            output='Above Average'
        elif(predicted_performance[0]==7):
            output=' Good'
        elif(predicted_performance[0]==8):
            output='Very Good'
        elif(predicted_performance[0]==9):
            output='Excellent'

        return output
    else:
        return 'Required form fields are missing'

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)