import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        '''
        param: X --> 2d array shape (120, 4)
        param: y --> target 1d array shape (120, )

        '''
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        '''
        param: x --> 2d arrays
        '''

        #Calculate Distance 
        # return a number: 1d array
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

        #Obtain K nearest samples, and labels
        # return index
        k_indices = np.argsort(distances)[:self.k]
        #return labels
        k_nearest_lables = [self.y_train[i] for i in k_indices]
        # Majority Vote
        # return List of Tuple
        common_label = Counter(k_nearest_lables).most_common(1)
        return common_label[0][0]






if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['red', 'blue', 'green'])

    iris = datasets.load_iris()
    X, y  = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 777)

    knn_clf = KNN(k = 3)
    knn_clf.fit(X_train, y_train)
    predictions = knn_clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)
    

    


