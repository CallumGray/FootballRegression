import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


premier_league:DataFrame = pd.read_json('Data/premier-league.json')
la_liga:DataFrame = pd.read_json('Data/la-liga.json')
bundesliga:DataFrame = pd.read_json('Data/bundesliga.json')
ligue_1:DataFrame = pd.read_json('Data/ligue-1.json')
serie_a:DataFrame = pd.read_json('Data/serie-a.json')

df:DataFrame = pd.concat([premier_league, la_liga, bundesliga, ligue_1, serie_a], axis=0, ignore_index=True)
df['statsbomb_pred'] = (df['statsbomb_xg'] > 0.5).astype(int)
df = df.sample(frac=1).reset_index(drop=True)

# Features for analysis
features = ['distance', 'angle', 'keeper_in_position', 'blocking_bodies', 'pressuring_players',
         'attacking_position', 'assist_ground', 'assist_low', 'assist_high', 'type_penalty',
         'type_freekick', 'technique_lob', 'technique_halfvolley', 'technique_volley',
         'technique_other', 'body_head', 'body_other', 'pattern_corner',
         'pattern_freekick', 'pattern_throw', 'pattern_counter']

target = 'goal'

# Train / Calibration / Test  splits
train_size:int = int(len(df) * 0.6)
cal_size:int = int(len(df) * 0.2)

train:DataFrame = df[:train_size]
cal:DataFrame = df[train_size:train_size + cal_size]
test:DataFrame = df[train_size + cal_size:]

# features
scaler = StandardScaler()
train_X:DataFrame = train[features]
cal_X:DataFrame = cal[features]
test_X:DataFrame = test[features]

# features scaled
train_X_scaled:ndarray = scaler.fit_transform(train_X) # (train_size, features)
cal_X_scaled:ndarray = scaler.transform(cal_X) # (cal_size, features)
test_X_scaled:ndarray = scaler.transform(test_X) # (test_size, features)

# target
train_y:ndarray = train[target].to_numpy() # (train_size,)
cal_y:ndarray = cal[target].to_numpy() # (cal_size,)
test_y:ndarray = test[target].to_numpy() # (test_size,)


# Logistic Regression model on the training data
uncalibrated_model:LogisticRegression = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced')
uncalibrated_model.fit(train_X_scaled, train_y)

# Use calibration data to get probabilities and calibrate
cal_uncalibrated:ndarray = uncalibrated_model.predict_proba(cal_X_scaled)[:, 1] # (cal_size,)
calibrator:IsotonicRegression = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(cal_uncalibrated, cal_y)


def plot_calibration():
    test_uncalibrated:ndarray = uncalibrated_model.predict_proba(test_X_scaled)[:, 1] # (test_size,)
    test_calibrated:ndarray = calibrator.transform(test_uncalibrated) # (test_size,)

    print("\nModel Performance on Test Set:")
    brier_uncalibrated:float = brier_score_loss(test_y, test_uncalibrated) # Uncalibrated
    brier_calibrated:float = brier_score_loss(test_y, test_calibrated) # Calibrated

    # Separate data into 10 bins based on their predicted probability (and take the mean for that bin as the x)
    # On the y axis, place the fraction of samples from that bin which were actually a goal
    fraction_goals_uncalibrated:list[float]
    mean_predicted_prob_uncalibrated:list[float]
    fraction_of_positives_uncalibrated, mean_predicted_prob_uncalibrated = calibration_curve(test_y, test_uncalibrated, n_bins=10)

    fraction_of_positives_calibrated:list[float]
    mean_predicted_value_calibrated:list[float]
    fraction_of_positives_calibrated, mean_predicted_value_calibrated= calibration_curve(test_y, test_calibrated, n_bins=10)

    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot( mean_predicted_prob_uncalibrated,fraction_of_positives_uncalibrated,"s-",label=f"Uncalibrated (Brier: {brier_uncalibrated:.3f})")
    plt.plot(mean_predicted_value_calibrated,fraction_of_positives_calibrated,"o-",label=f"Calibrated (Brier: {brier_calibrated:.3f})")
    plt.xlabel("Predicted probability of a goal (mean within bin)")
    plt.ylabel("Fraction of bin which are goals")
    plt.title("Calibration Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig("xg_calibration_plot_proper.png", dpi=300)
    plt.show()


def f1score_evaluation(predicted_y:ndarray, true_y:ndarray) -> float:

    true_positives:float = np.sum((predicted_y == 1) & (true_y == 1))
    false_positives:float = np.sum((predicted_y == 1) & (true_y == 0))
    true_negatives:float = np.sum((predicted_y == 0) & (true_y == 0))
    false_negatives:float = np.sum((predicted_y == 0) & (true_y == 1))

    precision:float = true_positives / (true_positives + false_positives)
    recall:float = true_positives / (true_positives + false_negatives)
    f1score:float = 2 * precision * recall / (precision + recall)

    return f1score


# Print out a map of the shot
def display_shot(shot_data:Series, xg:float=None) -> None:

    xg = round(xg, 3)
    statsbomb_xg = round(shot_data['statsbomb_xg'], 3)

    # Technique / Assist / Body
    technique:str = 'Normal'
    technique_map = {'technique_lob':'Lob','technique_halfvolley':'Half Volley','technique_volley':'Volley','technique_other':'Other'}
    for technique_key in technique_map.keys():
        if shot_data[technique_key] == 1:
            technique = technique_map[technique_key]

    assist:str = 'None'
    assist_map = {'assist_ground':'Ground', 'assist_low':'Low', 'assist_high':'High'}
    for assist_key in assist_map.keys():
        if shot_data[assist_key] == 1:
            assist = assist_map[assist_key]

    body:str = 'Foot'
    body_map = {'body_head':'Head', 'body_other':'Other'}
    for body_key in body_map.keys():
        if shot_data[body_key] == 1:
            body = body_map[body_key]

    metrics = 'xG: {}\nStatsbomb xG: {}\n\nAssist: {}\nBody Part: {}\nTechnique: {}'.format(xg, statsbomb_xg, assist, body, technique)

    # [x, y] or lists of [x, y]
    shot_location:list[float] = shot_data['shot_location']
    keeper_location:list[float] = shot_data['keeper_location']
    team_locations:list[list[float]] = shot_data['team_locations']
    op_locations:list[list[float]] = shot_data['op_locations']

    if keeper_location in op_locations:
        op_locations.remove(keeper_location)

    pitch_image = mpimg.imread('pitch.png')

    # Extracting x and y coordinates for each team
    shot_x, shot_y = [shot_location[0]], [shot_location[1]]
    keeper_x, keeper_y = [keeper_location[0]], [keeper_location[1]]
    team_x, team_y = [team_location[0] for team_location in team_locations], [team_location[1] for team_location in team_locations]
    op_x, op_y = [op_location[0] for op_location in op_locations], [op_location[1] for op_location in op_locations]

    plt.figure(figsize=(12, 8))
    plt.imshow(pitch_image, extent=(0., 120., 0., 80.))
    #plt.title('My xG: {}, Statsbomb xG: {}'.format(xg, statsbomb_xg))
    plt.text(125, 50, metrics, fontsize=20)
    plt.scatter(shot_x, shot_y, color='lime', label='Shot')
    plt.scatter(keeper_x, keeper_y, color='orange', label='GK')
    plt.scatter(team_x, team_y, color='blue', label='Team')
    plt.scatter(op_x, op_y, color='red', label='Opponents')

    plt.xlim(60, 120)
    plt.ylim(0, 80)
    plt.gca().invert_yaxis()
    plt.show()

# analyse a single shot (given as a 1 row dataframe)
def analyse_shot(model:LogisticRegression, calibrator:IsotonicRegression, shot_data:DataFrame, display:bool=False) -> float:

    shot_X:DataFrame = shot_data[features]
    shot_X_scaled:ndarray = scaler.transform(shot_X) # (1,features)
    uncalibrated_xg: ndarray = model.predict_proba(shot_X_scaled)[:,1] # (1,)
    calibrated_xg: ndarray = calibrator.transform(uncalibrated_xg) # (1,)
    xg:float = calibrated_xg[0]

    shot_series:Series = shot_data.iloc[0]
    if display:
        display_shot(shot_series, xg)

    return xg

def predict_n_samples(n:int, test:DataFrame) -> None:
    test_sample_indexes = np.random.randint(0, len(test), size=n)
    for index in test_sample_indexes:
        shot = test.iloc[index:index+1]
        analyse_shot(uncalibrated_model, calibrator, shot, True)

predict_n_samples(5, test)

test_probabilities_uncalibrated:ndarray = uncalibrated_model.predict_proba(test_X_scaled)[:,1]
test_probabilities_calibrated:ndarray = calibrator.transform(test_probabilities_uncalibrated)

test_brier_uncalibrated:float = brier_score_loss(test_y, test_probabilities_uncalibrated)
test_brier:float = brier_score_loss(test_y, test_probabilities_calibrated)

statsbomb_probabilities:ndarray = test['statsbomb_xg'].to_numpy()
statsbomb_brier:float = brier_score_loss(test_y, statsbomb_probabilities)

no_skill_predicted:ndarray = np.zeros((test_y.shape))
no_skill_brier:float = brier_score_loss(test_y, no_skill_predicted)

print('Brier Scores:')
print('Uncalibrated {}'.format(round(test_brier_uncalibrated,3)))
print('Calibrated {}'.format(round(test_brier,3)))
print('Statsbomb {}'.format(round(statsbomb_brier,3)))
print('No Skill {}'.format(round(no_skill_brier,3)))
