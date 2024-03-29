import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import logging
import json
from logging.handlers import RotatingFileHandler
from sklearn.preprocessing import LabelEncoder
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
    # output string for all prediction to send
    if all(key in request.form for key in ('job_profile_name_analysis', 'attained_questions_analysis', 'score_analysis', 'category_performance_analysis')):

        job_profile_name = request.form['job_profile_name_analysis']
        job_list = [job_profile_name]
        attained_questions_analysis = float(request.form['attained_questions_analysis'])
        # if attained_questions_analysis == 0:
        #     attained_questions_analysis = 1
        score_analysis = float(request.form['score_analysis'])
        print(attained_questions_analysis)
        print(attained_questions_analysis)
        correctness = round((score_analysis * 100) / attained_questions_analysis, 2)
        label_encoder = LabelEncoder()
        print(attained_questions_analysis)
        print(score_analysis)
        
        # Fit the LabelEncoder to the training data for job profiles
        if not hasattr(label_encoder, 'classes_'):
            label_encoder.fit(job_list)
        job_profile_name_analysis = label_encoder.transform([job_profile_name])[0]
        
        # Initialize output string
        output = f'Overall Performance for {job_profile_name} is '

        # Predict performance for the job profile
        data_to_predict = [attained_questions_analysis, score_analysis, correctness, job_profile_name_analysis]
        int_features = [float(x) for x in data_to_predict]
        final_features = [np.array(int_features)]
        predicted_performance = model.predict(final_features)

        # Map predicted performance to descriptive label
        performance_labels = ['Needs Improvement', 'Extremely Poor', 'Very Poor', 'Poor', 'Below Average', 'Average',
                              'Above Average', 'Good', 'Very Good', 'Excellent']
        job_profile_performance = performance_labels[predicted_performance[0]]
        output += f'{job_profile_performance} '
        output+='\n'
        output+=f'Skill Wise Analysis:'
     
        # Parse category data and predict performance for each category
        category_data = json.loads(request.form['category_performance_analysis'])
        for category_info in category_data:
            category_name = category_info['category']
            y_count = category_info['Y']
            n_count = category_info['N']
            total_count = category_info['total']
            category_list = [category_name]
            
            # Fit the LabelEncoder to the training data for categories
            if not hasattr(label_encoder, 'classes_'):
                label_encoder.fit(category_list)
            if category_name not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, category_name)
            
            category_name_analysis = label_encoder.transform([category_name])[0]
            
            # Calculate correctness for the category
            category_correctness = round((float(y_count) / float(total_count)) * 100, 2)
            
            # Predict performance based on correctness
            predicted_category_performance = model.predict([[total_count, y_count, category_correctness, category_name_analysis]])[0]
            category_performance_label = performance_labels[predicted_category_performance]
            
            # Append category performance to the output string
           
            output += f'\n {category_name} :{category_performance_label}, '

        # Remove trailing comma and space
        output = output[:-2]

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