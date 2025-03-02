import pandas as pd

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")

# Data preparation
y = df['logS']
x = df.drop('logS', axis=1)

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predictions
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_result = pd.DataFrame(["Linear regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_result.columns = ["methode","Training_MSE", "Training_R2_score", "testing_MSE", "Testing_R2_score"]

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Predictions
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# Evaluation
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_result = pd.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_result.columns = ["methode","Training_MSE", "Training_R2_score", "testing_MSE", "Testing_R2_score"]

# Combine results
final_results = pd.concat([lr_result, rf_result]).reset_index(drop=True)
print(final_results)