import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main():
    # Load the dataset
    data = pd.read_excel("Dataset502.xls", header=None)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels

    # kNN
    best_accuracy = 0
    best_k = 0
    for k in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        mean_accuracy = accuracies.mean()
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k

    print(f"Best k: {best_k}")
    print(f"Accuracy ΚΝΝ: {best_accuracy}")

    # Naïve Bayes classifier
    nb_classifier = GaussianNB()
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = cross_val_score(nb_classifier, X, y, cv=cv, scoring='accuracy')
    mean_accuracy = accuracies.mean()
    print("Naïve Bayes classifier assuming normal distribution:")
    print(f"Mean accuracy Naïve Bayes: {mean_accuracy}")

    # Testing different values of sigma for RBF kernel function
    best_accuracy = 0
    best_sigma = None
    for sigma in [0.1, 1, 10]:
        svm_classifier = SVC(kernel='rbf', gamma=1/(2 * sigma**2))
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = cross_val_score(svm_classifier, X, y, cv=cv, scoring='accuracy')
        mean_accuracy = accuracies.mean()
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_sigma = sigma

    print("Best value of sigma for RBF kernel function:", best_sigma)
    print("Mean accuracy RBF kernel function:", best_accuracy)

    # SVM classifier with linear kernel
    svm_classifier = SVC(kernel='linear')
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = cross_val_score(svm_classifier, X, y, cv=cv, scoring='accuracy')
    mean_accuracy = accuracies.mean()
    print("SVM with linear kernel:")
    print(f"Mean accuracy with linear kernel: {mean_accuracy}")

    # Different values for maximum tree depth and leaf size
    max_depth_values = [None, 10, 20, 30]
    min_samples_leaf_values = [1, 5, 10, 20]
    best_accuracy = 0
    best_max_depth = None
    best_min_samples_leaf = None

    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:
            dt_classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            accuracies = cross_val_score(dt_classifier, X, y, cv=cv, scoring='accuracy')
            mean_accuracy = accuracies.mean()
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_max_depth = max_depth
                best_min_samples_leaf = min_samples_leaf

    print("Decision Trees with different complexity levels:")
    print(f"Best accuracy Decision Trees: {best_accuracy}")
    print(f"Best max depth: {best_max_depth}")
    print(f"Best min samples leaf: {best_min_samples_leaf}")

if __name__ == "__main__":
    main()