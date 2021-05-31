import numpy as np

'''
Expected Input(X) Data Shape: ex (600, 30)
Expected Input(Y) Label Data Shape: ex (600, )

'''

class LinearRegress:

    def __init__(self, lr = 0.01, epochs = 100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_rows, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
                            #(80, 1) * ()
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_rows) * np.dot(X.T, (y_pred - y))
            db = (1 / n_rows ) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


def mean_squred_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state=32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1234)

    regress = LinearRegress()
    regress.fit(X_train, y_train)
    predicted = regress.predict(X_test)

    mse = mean_squred_error(y_test, predicted)
    print(mse)

    pred_line = regress.predict(X)
    cmap = plt.get_cmap()
    fig = plt.figure(figsize=(10, 10))
    m1 = plt.scatter(X_train, y_train, color = cmap(0.9), s = 10)
    m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s = 10)
    plt.plot(X, pred_line, label = 'Linear Regression Prediction')
    plt.show()




 

    