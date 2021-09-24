import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Lasso, Ridge
import seaborn as sns

data = pd.read_csv('flight_delay.csv')

# Airports One-Hot-Encoding

# creating new feature 'way', which way from departure to destination airport
data['way'] = data['Depature Airport'] + '-' + data['Destination Airport']

# encoding column 'way'
airports_way_dict = {}
count = 101
for i in data['way']:
    if i in airports_way_dict.keys():
        pass
    else:
        airports_way_dict[i] = count
        count += 1

# replacing str values of column way to int
dict_for_repalce_way = {'way': airports_way_dict}

data = data.replace(dict_for_repalce_way)

# len(data['Depature Airport'].unique()) # Output is '179'

# Assigning unique number to each airport
airports_dict = {}
count = 1
for i in data['Depature Airport']:
    if i in airports_dict.keys():
        pass
    else:
        airports_dict[i] = count
        count += 1

dict_for_repalce = {'Depature Airport': airports_dict}

# Replace airport names to unique numbers in Departure Airport column
data = data.replace(dict_for_repalce)
# In Destination Airport data we have one more airport code name
airports_dict['DME'] = 180

dict_for_repalce = {'Destination Airport': airports_dict}
# Replace airport names to unique numbers in Destination Airport column
data = data.replace(dict_for_repalce)

# Flight duration in minutes

data['Dep_datetime'] = pd.to_datetime(data['Scheduled depature time'])

data['Arrival_datetime'] = pd.to_datetime(data['Scheduled arrival time'])

q = data['Arrival_datetime'] - data['Dep_datetime']
# creating new column: flight duration
data['flight duration'] = q.dt.total_seconds().div(60).astype(int)

# Departure day of week

temp_arr = []


def date_info(dt):
    '''
    param dt: day, month and year
    return: day of week
    '''
    year, month, day = (int(x) for x in dt.split('-'))
    ans = datetime.date(year, month, day)
    temp_arr.append(ans.strftime("%A"))


for i in data['Dep_datetime']:
    i = str(i)
    date_info(i.split(' ')[0])
# new column day of week
data['day of week'] = pd.DataFrame(temp_arr)

days = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

day_of_week = {'day of week': days}
# replace str format of day of week to int
data = data.replace(day_of_week)

# is departure time was in the daytime or not
day_or_night = []
for i in data['Dep_datetime']:
    if 5 < int(str(i).split(' ')[1].split(':')[0]) < 22:
        day_or_night.append(1)
    else:
        day_or_night.append(0)

data['daytime'] = pd.DataFrame(day_or_night)

# SEASON
# In which season(winter, spring and etc.) plane flew
months = []
for i in data['Scheduled depature time']:
    if int(str(i).split(' ')[0].split('-')[1]) == 12 or int(str(i).split(' ')[0].split('-')[1]) <= 2:
        months.append(1)  # Winter
    elif 3 <= int(str(i).split(' ')[0].split('-')[1]) <= 5:
        months.append(2)  # Spring
    elif 6 <= int(str(i).split(' ')[0].split('-')[1]) <= 8:
        months.append(3)  # Summer
    elif 9 <= int(str(i).split(' ')[0].split('-')[1]) <= 11:
        months.append(4)  # Fall

data['season'] = pd.DataFrame(months)

# # Years

years = []
for i in data['Scheduled depature time']:
    years.append(i.split(' ')[0].split('-')[0])
data['years'] = pd.DataFrame(years)

# # Drop columns

data = data.drop(columns=['Scheduled depature time', 'Scheduled arrival time', 'Dep_datetime', 'Arrival_datetime'])

X = data.drop(columns='Delay')

# sns.pairplot(data[['Delay','season','day of week']])

# plt.scatter(data['flight duration'],data['Delay'])
# plt.title('Delay vs Flight duration')
# plt.ylabel('Delay')
# plt.xlabel('Flight Duration')

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)

# pca.fit(X)

# data_pca=pca.transform(X)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data_pca[:, 0],data['Delay'],data_pca[:,1])
# plt.title('Dataset after PCA')
#
# plt.show()


# # Split Data

test_data = data
data = data.drop(data[data.years.astype(int) > 2017].index)
test_data = test_data.drop(test_data[test_data.years.astype(int) <= 2017].index)

# Outliers
# detection outliers by z-score...
delay = data['Delay']
mean = np.mean(delay)
std = np.std(delay)

threshold = 2
outliers = []
out = []
for i in delay:

    z = (i - mean) / std  # formula of z-score
    if z > threshold:
        outliers.append(99999)  # if it outlier, so in new outliers column it will be '99999'
        out.append(i)
    else:
        outliers.append(i)

data['outliers'] = pd.DataFrame(outliers)

# data['outliers'].isnull().sum() # Output is '3'


# # Remove outliers

data = data.drop(data[data.outliers == 99999].index)
# data['outliers'].isnull().sum() # Output is still '3'
# data.tail() # last 3 rows with Nan in data['outliers']

# we removed outliers, but we still have another 3
data = data.drop([499164, 499168, 499535])

# data['outliers'].isnull().sum() # Output is '0'

data = data.drop(['outliers'], axis=1)

# # train data and test data

y_train = data['Delay']

data = data.drop(['Delay', 'years'], axis=1)

y_test = test_data['Delay']

X_test = test_data.drop(['Delay', 'years'], axis=1)

# Linear regression
print('Linear regression')
regression = LinearRegression()

lin_reg = regression.fit(data, y_train)

lin_reg_pred = lin_reg.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lin_reg_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, lin_reg_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lin_reg_pred)))

# # Polynomial regression on train data!

# train data
# degrees = [1, 2, 3]
#
# for i in degrees:
#
#     poly_feat = PolynomialFeatures(degree=i)
#     lin_reg = LinearRegression()
#     pipeline = Pipeline([("polynomial_features", poly_feat),
#                          ("linear_regression", lin_reg)])
#     pipeline.fit(data, y_train)
#
#
#     scores_mae = cross_val_score(pipeline, data, y_train, scoring="neg_mean_absolute_error", cv=8)
#     scores_mse = cross_val_score(pipeline, data, y_train, scoring="neg_mean_squared_error", cv=8)
#     scores_rmse = cross_val_score(pipeline, data, y_train, scoring="neg_root_mean_squared_error", cv=8)
#
#
#
#     print("Degree {}\nMAE = {}".format(i, -scores_mae.mean()),'STD: ', scores_mae.std())
#     print("Degree {}\nMSE = {}".format(i, -scores_mse.mean()),'STD: ', scores_mse.std())
#     print("Degree {}\nRMSE = {}".format(i, -scores_rmse.mean()),'STD: ', scores_rmse.std())

print('Polynomial Regression')
# test data
degrees = [1, 2, 3]

for i in degrees:
    poly_feat = PolynomialFeatures(degree=i)
    lin_reg = LinearRegression()
    pipeline = Pipeline([("polynomial_features", poly_feat),
                         ("linear_regression", lin_reg)])
    pipeline.fit(data, y_train)

    scores_mae = cross_val_score(pipeline, X_test, y_test, scoring="neg_mean_absolute_error", cv=5)
    scores_mse = cross_val_score(pipeline, X_test, y_test, scoring="neg_mean_squared_error", cv=5)
    scores_rmse = cross_val_score(pipeline, X_test, y_test, scoring="neg_root_mean_squared_error", cv=5)

    print("Degree {}\nMAE = {}".format(i, -scores_mae.mean()), 'STD: ', scores_mae.std())
    print("Degree {}\nMSE = {}".format(i, -scores_mse.mean()), 'STD: ', scores_mse.std())
    print("Degree {}\nRMSE = {}".format(i, -scores_rmse.mean()), 'STD: ', scores_rmse.std())

# Lasso

print('Lasso')
alphas = []
for i in range(15):
    alphas.append(1000 / (10 ** i))

MAE_list = []
MSE_list = []
RMSE_list = []

for i in alphas:
    lasso = Lasso(alpha=i)
    lasso.fit(data, y_train)
    prediction = lasso.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, prediction)
    mse = metrics.mean_squared_error(y_test, prediction)
    rmse = math.sqrt(metrics.mean_squared_error(y_test, prediction))

    MAE_list.append(mae)
    MSE_list.append(mse)
    RMSE_list.append(rmse)

best_alpha_mae = alphas[np.argmin(MAE_list)]
min_arg_mae = min(MAE_list)

index = alphas.index(best_alpha_mae)

min_arg_mse = MSE_list[index]
min_arg_rmse = RMSE_list[index]

print("Best value of alpha:", best_alpha_mae, '\nMAE= ', min_arg_mae,
      '\nMSE= ', min_arg_mse, '\nRMSE= ', min_arg_rmse)

# Ridge
print('Ridge')
MAE_list = []
MSE_list = []
RMSE_list = []
for i in alphas:
    lasso = Ridge(alpha=i)
    lasso.fit(data, y_train)
    prediction = lasso.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, prediction)
    mse = metrics.mean_squared_error(y_test, prediction)
    rmse = math.sqrt(metrics.mean_squared_error(y_test, prediction))

    MAE_list.append(mae)
    MSE_list.append(mse)
    RMSE_list.append(rmse)

best_alpha_mae = alphas[np.argmin(MAE_list)]
min_arg_mae = min(MAE_list)

index = alphas.index(best_alpha_mae)

min_arg_mse = MSE_list[index]
min_arg_rmse = RMSE_list[index]

print("Best value of alpha:", best_alpha_mae, '\nMAE= ', min_arg_mae,
      '\nMSE= ', min_arg_mse, '\nRMSE= ', min_arg_rmse)
