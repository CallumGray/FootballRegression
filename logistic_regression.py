import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):

            linear_combo = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_combo)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Descend
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        # Predict probability of class = 1
        linear_combo = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_combo)

    def predict(self, X, threshold=0.5):
        # Predict probability for all samples
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)



premier_league_df = pd.read_json('Data/premier-league.json')
la_liga_df = pd.read_json('Data/la-liga.json')
bundesliga_df = pd.read_json('Data/bundesliga.json')
ligue_1_df = pd.read_json('Data/ligue-1.json')
serie_a_df = pd.read_json('Data/serie-a.json')

df = pd.concat([premier_league_df, la_liga_df, bundesliga_df, ligue_1_df, serie_a_df], axis=0, ignore_index=True)
df = df.sample(frac=1, random_state=1000).reset_index(drop=True)

train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

'''
'shot_location','keeper_location','team_locations','op_locations','statsbomb_xg'
'''

xcols = ['distance','angle','keeper_in_position','blocking_bodies','pressuring_players','attacking_position','assist_ground','assist_low','assist_high','type_penalty','type_freekick','technique_lob','technique_halfvolley','technique_volley','technique_other','body_head','body_other','pattern_corner','pattern_freekick','pattern_throw','pattern_counter']

train_X = train_df[xcols]
train_y = train_df['goal']

test_X = test_df[xcols]
test_y = test_df['goal']

# Train
model = LogisticRegression()
model.fit(train_X, train_y)

pred_test_y = model.predict(test_X)

correct = np.sum(pred_test_y == test_y)
total = len(test_y)
print('correct', correct)
print('total', total)
accuracy = correct / total
print('accuracy', accuracy)

print(accuracy)
