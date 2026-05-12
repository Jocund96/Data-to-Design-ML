from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the California Housing dataset
housing = fetch_california_housing()

# Describe the dataset
print(housing.DESCR)

# Set a seed to split the data 
h_X_train, h_X_test, h_Y_train, h_Y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=0)
# h_X_train, h_X_test, h_Y_train, h_Y_test = train_test_split(housing.data[:,[0,5,6]], housing.target, test_size=0.2, random_state=0)

# Fit a model(s)
lin_reg = LinearRegression().fit(h_X_train, h_Y_train)

# Prediction
h_Y_pred = lin_reg.predict(h_X_test)

# Report relevant metrics
r_square = r2_score(h_Y_test, h_Y_pred)
mae = mean_absolute_error(h_Y_test, h_Y_pred)

print(f"R2 score = {r_square}")
print(f"MAE =  {mae}")

# Calculate the train R2
h_Y_pred_tr = lin_reg.predict(h_X_train)
r_square_tr = r2_score(h_Y_train, h_Y_pred_tr)
print(f"Train R2 score = {r_square_tr}")

# Actual vs predicted values
plt.scatter(h_Y_test, h_Y_pred)
plt.plot([h_Y_test.min(), h_Y_test.max()], [h_Y_test.min(), h_Y_test.max()], "r")
plt.show()