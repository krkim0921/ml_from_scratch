import numpy as np


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

class LogisticRegress:

    def __init__(self, lr = 0.001, epochs = 100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_rows, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradients Decent
        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(model)
            
            dw = (1 / n_rows) * np.dot(X.T, (y_pred - y))
            db = (1 / n_rows) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr *db

    
    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(model)
        pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return pred_cls
   
    def _sigmoid(self, x):
        return 1 / (1+ np.exp(-x))

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 777)

    logistic_regress = LogisticRegress(lr = 0.001, epochs= 1000)
    logistic_regress.fit(X_train, y_train)
    pred = logistic_regress.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, pred))





