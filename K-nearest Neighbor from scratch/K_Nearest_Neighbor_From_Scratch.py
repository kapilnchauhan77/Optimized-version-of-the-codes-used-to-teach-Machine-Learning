import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib import style
import warnings

style.use('fivethirtyeight')

dataset = {"k": [[1, 2], [2, 3], [3, 1]], "r": [[6, 5], [7, 7], [8, 6]]}


def k_nearest_neighbor(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is less than then size of your data groups!!!')

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

    return vote_result


result = k_nearest_neighbor(dataset, [5, 7])

print(f"Group: {result}")

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(5, 7, s=200, color=result, marker='*')
plt.show()
