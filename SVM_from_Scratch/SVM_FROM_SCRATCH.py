import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data

        opt_dict = {}

        transforms = [[1 for _ in range(len(list(self.data.values())[0][0]))]]
        cnt = 0
        for length in range(len(list(self.data.values())[0][0])):
            for j in range(len(list(self.data.values())[0][0])):
                a = [1 for _ in range(len(list(self.data.values())[0][0]))]
                for k in range(cnt):
                    try:
                        a[j+k] = -1
                    except:
                        pass
                    transforms.append(a)
                cnt += 1

        for idx, i in enumerate(transforms):
            if i in transforms[idx+1:]:
                transforms.pop(idx)

        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001, ]

        b_range_multiple = 3
        b_multiple = 5
        latest_optimum = self.max_feature_value

        for step in step_sizes:
            w = np.array([latest_optimum for i in range(
                len(list(self.data.values())[0][0]))])

            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   (self.max_feature_value * b_range_multiple),
                                   step * b_multiple):

                    for transformations in transforms:
                        w_t = w * transformations
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break

                    if found_option:
                        opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * 2
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(f"{xi}: {yi * np.dot(self.w, xi) + self.b}")

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200,
                            marker="*", color=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
          for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {
    -1: np.array([
        [1, 7],
        [2, 8],
        [3, 8]
    ]),

    1: np.array([
        [5, 1],
        [6, -1],
        [7, 3]
    ])}

to_predict = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

for p in to_predict:
    svm.predict(p)

svm.visualize()


# [[plt.scatter(ii[0], ii[1], s=200, c='r' if i == 1 else 'b', marker="*")
#   for ii in data_dict[i]] for i in data_dict]
# plt.show()
