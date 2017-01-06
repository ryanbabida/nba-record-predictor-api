#!flask/bin/python
from flask import Flask, jsonify, abort, request
from flask_cors import CORS, cross_origin
from data_processing.data_cleaning import *
from data_processing.grad_descent import *
from data_processing.ols import *
import numpy as np
import random
import sys, os

app = Flask(__name__)
CORS(app)


@app.route('/api/v1.0/weights', methods=['GET', 'POST'])
def weights():
    ols_weights = list()
    gd_weights = list()
    features, target, names = load_data()
    b = ols(features, target)
    b = b.tolist()

    w, e = gradient_descent(0.02, features, target, features.shape[0], 50)

    w = w.tolist()

    if len(b) == 0 or len(w) == 0 or len(features) == 0: 
        abort(404)

    for i in range(len(b)): 
        ols_weights.append(b[i][0])
        gd_weights.append(w[i][0])

    return jsonify({
        'names': names[4:26],
        'ols_weights': ols_weights, 
        'gd_weights': gd_weights,
        'gd_err': e,
        })


@app.route("/api/v1.0/predict", methods=['GET'])
def makePrediction():
    features_sample = list()
    keys = ['MIN', 'PTS', 'FGM', 'FGA', 'FG', '3PM', '3PA', 
                '3P', 'FTM', 'FTA', 'FT', 'OREB', 'DREB', 
                'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 
                'PF', 'PFD', 'MAR']
    for key in keys:
        if not request.args.get(key):
            abort(404)
        num = float(request.args.get(key))
        features_sample.append(num)

    if len(features_sample) != 22: 
        abort(404)

    features_sample = np.matrix(features_sample)

    features, target, names = load_data()
    b = ols(features, target)
    w, e = gradient_descent(0.02, features, target, features.shape[0], 50)
    gd_prediction = predict(features_sample, w)
    gd_prediction = gd_prediction.tolist()
    ols_prediction = predict(features_sample, b)
    ols_prediction = ols_prediction.tolist()

    if len(b) == 0 or len(features) == 0: 
        abort(404)

    return jsonify({
        'ols_prediction': ols_prediction[0],
        'gd_prediction': gd_prediction[0]
        })


@app.route("/api/v1.0/sample", methods=['GET'])
def getDataSample():
    features_sample = list()
    ols_predictions = list()
    gd_predictions = list()
    _target = list()

    features, target, names = load_data()
    b = ols(features, target)
    w, e = gradient_descent(0.02, features, target, features.shape[0], 50)


    features, target, names, team_names, games_played, wins = load_test()

    if len(b) == 0 or len(w) == 0 or len(features) == 0: 
        abort(404)

    for i in range(len(features)):
        ols_prediction = predict(features[i], b)
        ols_prediction = ols_prediction.tolist()
        ols_predictions.append(round(ols_prediction[0][0], 2))
        gd_prediction = predict(features[i], w)
        gd_prediction = gd_prediction.tolist()
        gd_predictions.append(round(gd_prediction[0][0], 2))

    test_set = features.tolist()
    target = target.tolist()
    games_played = games_played.tolist()
    wins = wins.tolist()

    for i in range(len(target)): 
        test_set[i].append(target[i][0])
        test_set[i].append(ols_predictions[i])
        test_set[i].append(gd_predictions[i])

    test_set = [[str(float(j)) for j in i] for i in test_set]

    for i in range(len(target)): 
        _target.append(target[i][0])

    test_set.insert(0, ['TEAM', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', 
             '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 
             'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 
             'PF', 'PFD', '+/-', 'ACT W%', 'OLS W%', 'GD W%'])

    results = list()

    for i in range(len(team_names)): 
        test_set[i + 1].insert(0, team_names[i])
        results.append([team_names[i], games_played[i], wins[i], games_played[i] - wins[i], _target[i], int(games_played[i] * ols_predictions[i]), 
            games_played[i] - int(games_played[i] * ols_predictions[i]), int(games_played[i] * gd_predictions[i]), 
            games_played[i] - int(games_played[i] * gd_predictions[i])])

    results.insert(0, ['TEAM', 'GP', 'W', 'L', 'W%', 'OLS-W', 'OLS-L', 'GD-W', 'GD-L'])

    return jsonify({
        'test_set': test_set,
        'target': _target,
        'ols_predictions': ols_predictions,
        'gd_predictions': gd_predictions,
        'results': results,
        })


if __name__ == '__main__':
    app.run(debug=True)