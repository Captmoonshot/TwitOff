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

    regressor = pickle.load(open(
        os.path.join('pkl_objects',
            'regressor.pkl'), 'rb'))

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



    @app.route('/reset')
    def reset():
    	DB.drop_all()
    	DB.create_all()
    	return render_template('base.html', title='DB Reset!', users=[])

    return app





