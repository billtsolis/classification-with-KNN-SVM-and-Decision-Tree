In the present report, the construction and evaluation of four different classifiers were examined using a two-class classification dataset. The classifiers that were studied are as follows:

  1    kNN (Nearest Neighbor Classifier) with various values of kk.
  2    Naïve Bayes Classifier with a normal distribution.
  3    SVM (Support Vector Machines) Classifier with an RBF kernel function and a linear kernel.
  4    Decision Trees Classifier, adjusting the complexity of the tree in various ways (e.g., number of nodes or leaf size).

To evaluate the generalization of the classifiers, the 10-fold cross-validation method was applied. The dataset was divided into 10 groups with an equal number of samples, and 10 repeated experiments were conducted. In each iteration, the classifier was trained on 9 groups and evaluated on the remaining group.
