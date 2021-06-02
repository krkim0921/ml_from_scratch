import numpy as np
from scipy.sparse import data


def accuracy(label, y_pred):
    accuracy = np.sum(label == y_pred) / len(label)
    return accuracy


class Perceptron:

    def __init__(self, lr = 0.001, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.activation = self._unit_step
        self.weights = None
        self.bias = None

    
    def fit(self, X, y):
        '''
        param X: ndarray,

        '''
        n_rows, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        label = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                lin_ouput = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(lin_ouput)
                update = self.lr * (label[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(output)
        return y_pred




    def _unit_step(self, x):
        return np.where(x >=0, 1, 0)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2,
                              cluster_std= 1.05, random_state=777)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 777)

    classifier = Perceptron(lr = 0.001, epochs = 1000)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    print(f'Classfication Accuracy: {accuracy(y_test, pred)}')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker = 'o', c = y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-classifier.weights[0] * x0_1 - classifier.bias) / classifier.weights[1]
    x1_2 = (-classifier.weights[0] * x0_2 - classifier.bias) / classifier.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin-3, ymax + 3])

    plt.show()


