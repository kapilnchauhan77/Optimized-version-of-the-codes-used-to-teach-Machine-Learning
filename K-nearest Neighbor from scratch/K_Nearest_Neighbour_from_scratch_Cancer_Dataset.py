import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop('id', inplace=True, axis=1)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2

train_set = {4: [], 2: []}
test_set = {4: [], 2: []}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]


def k_nearest_neighbor(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is less than number of data groups!!!')

    distances = []

    for group in data:
        for features in data[group]:
            # Euqlidian_Distance = ((features[0] - predict[0]) ** 2 + (features[1] - predict[1]) ** 2) ** 0.5
            # Numpy(Better)_Euqlidian_Distance = np.sqrt(np.sum(np.square(np.array(features) - np.array(predict))))
            Final_Euqlidian_Distance = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append(
                [Final_Euqlidian_Distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    Confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, Confidence


accuracies = []
Confidences = []

for i in range(10):
    correct = 0
    total = 0

    confidences = []

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbor(train_set, data, k=5)
            confidences.append(confidence)
            if vote == group:
                correct += 1
            else:
                pass
                # print(confidence)

            total += 1

    acc = correct / total
    confi = np.mean(confidences)
    accuracies.append(acc)
    Confidences.append(confi)
print(f"Final Accuracy: {round(np.mean(accuracies)*100, 2)}%")
print(f"Final Confidence: {round(np.mean(Confidences)*100, 2)}%")
