import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_excel('titanic.xls')

df.drop(['name', 'body'], axis=1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna('0', inplace=True)

# df = pd.get_dummies(df, columns=['sex', 'cabin',
# 'embarked', 'home.dest'], drop_first = True)
# ohe = preprocessing.OneHotEncoder(sparse=False)
# column_trans = make_column_transformer((ohe, ['sex',
#                                               'cabin',
#                                               'embarked',
#                                               'home.dest']), remainder='passthrough')
# df['sex', 'cabin', 'embarked', 'home.dest'] = df[['sex',
#                                                   'cabin',
#                                                   'embarked',
#                                                   'home.dest', ]].apply(le.fit_transform)
# df = ohe.fit_transform(df)
# df = column_trans.fit_transform(df)
# x = np.array(df[:-1].astype(float))
# y = np.array(df[-1].astype(float))


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


# df = handle_non_numerical_data(df)


# df.drop(['embarked', 'home.dest'], 1, inplace=True)
print(df.head())
# X = np.array(df.drop(['survived'], 1).astype(float))
X = df.drop(['survived'], 1)
X.drop(['pclass', 'age', 'sibsp', 'parch', 'ticket', 'fare',
        'boat'], 1, inplace=True)
print(X.columns)
X = pd.get_dummies(df, columns=['sex', 'cabin', 'embarked', 'home.dest'], drop_first=True)

X = np.array(X)
X = X[:, 8:].astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
