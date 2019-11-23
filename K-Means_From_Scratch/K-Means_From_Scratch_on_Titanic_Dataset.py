import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import style
style.use('fivethirtyeight')


class KMeans(object):
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid])
                             for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


df = pd.read_excel('titanic.xls')

df.drop(['name', 'body'], axis=1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna('0', inplace=True)

# df = pd.get_dummies(df, columns=['sex', 'cabin',
#                                  'embarked', 'home.dest'], drop_first=True)
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


df = handle_non_numerical_data(df)


df.drop(['embarked', 'home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)

y = np.array(df['survived'])
clf = KMeans()
clf.fit(X)

colors = ["g", "r", "c", "b", "k", "o", "y"]

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print(correct/len(X))
