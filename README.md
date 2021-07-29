# kaggle_mlb
Kaggle competition repository


#### OLD
1. Expanding Mean, Median, Median(365 days) - 1.614
2. 1 + lag1-3 statusCode, teamid, change status code 2-4 - 1.553
3. 2 + dayofweek - 1.561
4. 2 + tr_scores2 - lag1 - 1.475
5. 4 + lag 2 - 1.460
6. 5 + tr_scores - lag1 - 1.465
7. 4 + lastnsum (2, 7, 30, 90) - 1.456 

#### Final Dash:
* Improve single model
    * Add game type, current team standing and opposing team standing
    * transaction to and from teams
    * avg velocity, angle and distance
    * final team1 score, team2 score, std in diff of scores at each innings end
    * other statcast features

* Rigorous validation
    * Check score on June, July (Impact of changing windows, cleaning of featuresm hyperparams)
    * Impact of adding more training data on July validation

* Look out for august specific things
    * 2019 - July score v/s august score ???





