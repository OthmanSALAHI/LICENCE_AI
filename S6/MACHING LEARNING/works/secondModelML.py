import pandas as pd

# matches = pd.read_csv(r'C:\Users\Othman SALAHI\Desktop\STUDY\MACHING LEARNING\data\matches.csv',index_col=0)
matches = pd.read_csv("https://raw.githubusercontent.com/dataquestio/project-walkthroughs/refs/heads/master/football_matches/matches.csv")
# print(matches.head())


# fouille de données
matches['date'] = pd.to_datetime(matches['date'])
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(':.+', '', regex=True).astype("int")
matches['day_code'] = matches['date'].dt.dayofweek
matches["target"] = (matches["result"] == 'W').astype("int")


# creating the model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches[matches['date'] < '2022-01-01']
test = matches[matches['date'] >= '2022-01-01']

predictors = ["venue_code", "opp_code", "hour", "day_code"]

rf.fit(train[predictors], train['target'])

preds = rf.predict(test[predictors])

# evaluate the performance

from sklearn.metrics import accuracy_score

acc = accuracy_score(test["target"], preds)

combined = pd.DataFrame(dict(actual=test['target'], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined['prediction'])

from sklearn.metrics import precision_score

precision_score(test['target'], preds)

# adding more predictors

grouped_matches = matches.groupby('team')

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f"{c}_rolling" for c in cols]

# example of rolling average

# group = grouped_matches.get_group('Manchester City')
# rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

matches_rolling = matches_rolling.droplevel("team")

matches_rolling.index = range(matches_rolling.shape[0])

def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

combined, error = make_predictions(matches_rolling, predictors + new_cols)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

import numpy as np
combined['result_predicted'] = np.where(combined['predicted'] == 1, 'W', 'L or D')
combined

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", 
              "Manchester United": "Manchester Utd", 
              "Newcastle United": "Newcastle Utd", 
              "Tottenham Hotspur": "Tottenham", 
              "West Ham United": "West Ham", 
              "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

# merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()

print(merged)


