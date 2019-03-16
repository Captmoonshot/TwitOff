"""Main application and routing logic for TwitOff."""
from flask import Flask, render_template, request
from decouple import config
from .models import DB, User, Tweet
from .twitter import add_or_update_user
from .predict import predict_user

# Imports for make_predict()
from flask import abort, jsonify, request
import os
import pickle
import numpy as np



def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # app.config['FLASK_ENV'] = config('FLASK_ENV')
    DB.init_app(app)
    # My RandomForestRegressor for Flask API
    regressor = pickle.load(open(
        os.path.join('pkl_objects',
            'regressor.pkl'), 'rb'))
    # Manjula's RandomForestRegressor for Flask API
    m = pickle.load(open(
        os.path.join('pkl_objects',
            'zillow.pkl'), 'rb'))


    @app.route('/')
    def root():
        users = User.query.all()
        #tweets = Tweet.query.all()
        #return render_template('base.html', title='Home', users=users, tweets=tweets)
        return render_template('base.html', title='Home', users=users)

    @app.route('/user', methods=['POST'])
    @app.route('/user/<name>', methods=['GET'])
    def user(name=None):
        message = ''
        #import pdb; pdb.set_trace()
        name = name or request.values['user_name']
        try:
            if request.method == 'POST':
                add_or_update_user(name)
                message = 'User {} successfully added!'.format(name)
            tweets = User.query.filter(User.name == name).one().tweets
            
        except Exception as e:
            message = 'Error adding {}:{}'.format(name, e)
            tweets = []
        return render_template('user.html', title=name, tweets=tweets,
            message=message)

    @app.route('/compare', methods=['POST'])
    def compare():
        user1, user2 = request.values['user1'], request.values['user2']
        tweeted = request.values['tweet_text']
        if user1 == user2:
            return 'Cannot compare a user to themselves!'
        else:
            y, proba = predict_user(user1, user2,
                                    request.values['tweet_text'])
            

            # return user1 if y else user2
            if y:
                return render_template('compare.html',
                                    user=user1,
                                    tweeted=tweeted,
                                    prediction=y,
                                    probability=round(proba*100, 2))
            else:
                return render_template('compare.html',
                                    user=user2,
                                    tweeted=tweeted,
                                    prediction=y,
                                    probability=round(proba*100, 2))

    

    @app.route('/api', methods=['POST'])
    def make_predict():
        data = request.get_json(force=True)
        predict_request = [data['beds'], data['baths'], data['square_feet'], data['lot_size'], data['hoa_per_month'], data['property_types'], data['property_age'], data['zip_codes']]
        predict_request = np.array(predict_request)
        y_hat = regressor.predict([predict_request])
        y_hat = np.round(y_hat)
        output = y_hat[0]
        return jsonify(results=output)

    @app.route('/manjula', methods=['POST'])
    def estimate_value():
        json_data = request.get_json(force=True)
        house_features = [json_data['year_assessment'],json_data['land_use_type'], json_data['beds'], 
                        json_data['baths'], json_data['total_rooms'], json_data['zip'], json_data['assessed_property_taxes'], 
                        json_data['year_built'], json_data['sqft_house']]
        house_features = np.array(house_features)
        y_hat = m.predict([house_features])
        y_hat = np.round(y_hat)
        output = y_hat[0]
        return jsonify(results=output)
        
    """To make the API call from python or Jupyter Notebook:
    import requests
    import json

    #url = 'http://127.0.0.1:5000/manjula'
    url = 'https://captmoonshot-twitoff.herokuapp.com/manjula'
    data = json.dumps({'year_assessment':2015, 'land_use_type':261, 'beds':7, 'baths':2.0, 'total_rooms':0, 'zip':90210, 'assessed_property_taxes':3402.94, 'year_built': 1963, 'sqft_house':2000.0})
    r = requests.post(url, data)

    print(r.json())
    """




    @app.route('/reset')
    def reset():
    	DB.drop_all()
    	DB.create_all()
    	return render_template('base.html', title='DB Reset!', users=[])

    return app





