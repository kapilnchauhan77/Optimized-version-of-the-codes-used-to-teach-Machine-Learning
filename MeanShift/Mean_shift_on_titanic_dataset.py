import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing


'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''


df = pd.read_excel('titanic.xls')

df.drop(['body', 'name'], 1, inplace=True)
original_df = pd.DataFrame.copy(df)
df.fillna(0, inplace=True)

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


df = handle_non_numerical_data(df)


df.drop(['embarked', 'home.dest'], 1, inplace=True)
# print(df.head())
X = np.array(df.drop(['survived'], 1).astype(float))


# X = df.drop(['survived'], 1)
# X.drop(['pclass', 'age', 'sibsp', 'parch', 'ticket', 'fare',
#         'boat'], 1, inplace=True)
# print(X.columns)
# X = pd.get_dummies(df, columns=['sex', 'cabin', 'embarked', 'home.dest'], drop_first=True)

# X = np.array(X)
# X = X[:, 8:].astype(float)


X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan


for idx, i in enumerate(labels):
    original_df['cluster_group'].iloc[idx] = i

n_clusters = len(np.unique(labels))

survival_rates = {}
print(n_clusters)
for i in range(n_clusters):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = (len(survival_cluster) / len(temp_df))
    survival_rates[i] = f"{round(survival_rate*100, 2)}%"
print(survival_rates)
original_df.fillna(0, inplace=True)
original_df.drop(['cabin', 'home.dest'], 1, inplace=True)
# ohe = preprocessing.OneHotEncoder(sparse=False)
# original_df['sex'] = ohe.fit_transform(original_df[['sex']])
original_df = pd.get_dummies(original_df, columns=['sex'], drop_first=True)
female = []
for i in original_df['sex_male']:
    female.append(abs(i-1))
original_df['sex_female'] = np.array(female)
# original_df = handle_non_numerical_data(original_df)
# print(original_df.head())
# print(original_df.columns)
for i in range(n_clusters):
    print(f"Data for cluster: {i}")
    print(original_df[(original_df['cluster_group'] == float(i))].describe())
