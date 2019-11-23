import pandas as pd
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop('id', inplace=True, axis=1)

x = np.array(df.drop('class', axis=1))
y = np.array(df['class'])

# x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
# clf = svm.SVC(kernel='poly')
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test)

print(f"Accuracy: {round(acc*100, 2)}%")

to_predict = np.array([[10, 7, 7, 6, 4, 10, 4, 4, 2], [
                      5, 4, 4, 5, 7, 2, 3, 2, 1]])

to_predict = to_predict.reshape(len(to_predict), -1)

predictions = [i for i in clf.predict(to_predict)]

for j in predictions:
    print("######################\n Pediction: Benign \n Real:      Benign") if j == 2 else print(
        "####################\n Pediction: Malignant \n Real:      Malignant")
