import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import style
import matplotlib.pyplot as plt
import datetime
import pickle as pkl

style.use('ggplot')

quandl.ApiConfig.api_key = "QXMHf2yZnzTcgRJJa5ha"
df = quandl.get('WIKI/GOOGL')
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100


forecast_col = 'Adj. Close'
df = df[["Adj. Close", "HL_PCT","PCT_change", "Adj. Volume"]]
df.fillna(-99999, inplace=True)
forecast_out = math.ceil(0.01 * len(df))
df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

print(f"Dataframe size: {len(df)}")

x = np.array(df.drop(['label'], axis = 1))
y = np.array(df['label'])

x = preprocessing.scale(x)

x_lately = x[-forecast_out:]
x = x[:-forecast_out]

y_lately = y[-forecast_out:]
y = y[:-forecast_out]  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

logreg = LinearRegression()

logreg.fit(x_train, y_train)

support_vector_machine = svm.SVR(kernel = 'rbf')

support_vector_machine.fit(x_train, y_train)

gpr = GaussianProcessRegressor()
gpr.fit(x_train, y_train)

rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

print(f"Score of linear regression: {logreg.score(x_test, y_test)}")
with open('LinearRegression.pkl', 'wb') as f:
	pkl.dump(logreg, f)
print(f"Score of support vector machine: {support_vector_machine.score(x_test, y_test)}")
with open('support_vector_machines.pkl', 'wb') as f:
	pkl.dump(support_vector_machine, f)
print(f"Score of Gaussian Process regressor: {gpr.score(x_test, y_test)}")
with open('GaussianProcessregressor.pkl', 'wb') as f:
	pkl.dump(gpr, f)
print(f"Score of Random Forest: {rfr.score(x_test, y_test)}")
with open('RandomForestRegressor.pkl', 'wb') as f:
	pkl.dump(rfr, f)
print(f"Score of Decision Tree: {dt.score(x_test, y_test)}")
with open('DecisionTreeRegressor.pkl', 'wb') as f:
	pkl.dump(dt, f)

forecast_set = logreg.predict(x_lately)

print(f"Predicted prices: {forecast_set}, accuracy: {rfr.score(x_test, y_test)}")

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 24 * 60 * 60 
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
