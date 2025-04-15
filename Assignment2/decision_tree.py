# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def partition(x):
    unique_values = np.unique(x)
    partitions = {v: np.where(x == v)[0] for v in unique_values}
    return partitions

def entropy(y):
    unique_values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def binary_information_gain(x, y):
    n = len(x)
    p_true = np.sum(x) / n
    p_false = 1 - p_true
    H_y = entropy(y)
    H_true = entropy(y[x]) if np.sum(x) > 0 else 0
    H_false = entropy(y[~x]) if np.sum(~x) > 0 else 0
    return H_y - (p_true * H_true + p_false * H_false)

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(x.shape[1]):
            for value in np.unique(x[:, i]):
                attribute_value_pairs.append((i, value))
                
    unique_labels = np.unique(y)
    if len(unique_labels) == 1:
        return unique_labels[0]
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return np.bincount(y).argmax()
    
    max_info_gain = -np.inf
    best_pair = None
    for pair in attribute_value_pairs:
        feature_index, candidate_value = pair
        condition = (x[:, feature_index] == candidate_value)
        info_gain = binary_information_gain(condition, y)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_pair = pair

    if max_info_gain <= 0:
        return np.bincount(y).argmax()
    
    remaining_pairs = [pair for pair in attribute_value_pairs if pair != best_pair]
    feature_index, candidate_value = best_pair

    condition = (x[:, feature_index] == candidate_value)
    true_indices = np.where(condition)[0]
    false_indices = np.where(~condition)[0]

    tree = {}
    if len(true_indices) == 0:
        tree[(feature_index, candidate_value, True)] = np.bincount(y).argmax()
    else:
        tree[(feature_index, candidate_value, True)] = id3(x[true_indices], y[true_indices], remaining_pairs, depth+1, max_depth)
    
    if len(false_indices) == 0:
        tree[(feature_index, candidate_value, False)] = np.bincount(y).argmax()
    else:
        tree[(feature_index, candidate_value, False)] = id3(x[false_indices], y[false_indices], remaining_pairs, depth+1, max_depth)
    
    return tree

def predict_example(x, tree):
    for key, subtree in tree.items():
        feature_index, candidate_value, branch_condition = key
        if (x[feature_index] == candidate_value) == branch_condition:
            if isinstance(subtree, dict):
                return predict_example(x, subtree)
            else:
                return subtree
    return None

def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

def visualize(tree, depth=0):
    if depth == 0:
        print('TREE')
    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))
        if isinstance(sub_trees, dict):
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def confusion_matrix(y_true, y_pred):
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])

def print_predictions(X, y, tree):
    """
    For each example in X, print its features, true label from y, and the predicted label.
    
    Parameters:
    - X: numpy array of features (each row is an example)
    - y: numpy array of true labels corresponding to the rows in X
    - tree: decision tree produced by the id3 function
    """
    for i in range(len(X)):
        predicted = predict_example(X[i], tree)
        print(f"Row {i} Features: {X[i]}  |  True Label: {y[i]}  |  Predicted: {predicted}")

if __name__ == '__main__':
    # Load MONK's problems data
    monk_problems = ['monks-1', 'monks-2', 'monks-3']
    depths = range(1, 11)
    for problem in monk_problems: #FOR ALL # MONKS
        train_file = f'./data/{problem}.train'
        test_file = f'./data/{problem}.test'
        M_train = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        y_train = M_train[:, 0]
        X_train = M_train[:, 1:]
        M_test = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        y_test = M_test[:, 0]
        X_test = M_test[:, 1:]
        train_errors = []
        test_errors = []
        for depth in depths: #FOR ALL DEPTHS 1-10
            decision_tree = id3(X_train, y_train, max_depth=depth)
            y_train_pred = [predict_example(x, decision_tree) for x in X_train]
            train_err = compute_error(y_train, y_train_pred)
            train_errors.append(train_err)
            y_test_pred = [predict_example(x, decision_tree) for x in X_test]
            test_err = compute_error(y_test, y_test_pred)
            test_errors.append(test_err)
        plt.figure()
        plt.plot(depths, train_errors, label='Training Error')
        plt.plot(depths, test_errors, label='Test Error')
        plt.xlabel('Tree Depth')
        plt.ylabel('Error')
        plt.title(f'Learning Curves for {problem}')
        plt.legend()
        plt.show()

    # Weak Learners for monks-1
    M_train = np.genfromtxt('./data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y_train = M_train[:, 0]
    X_train = M_train[:, 1:]
    M_test = np.genfromtxt('./data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y_test = M_test[:, 0]
    X_test = M_test[:, 1:]
    decision_tree_depth1 = id3(X_train, y_train, max_depth=1)
    visualize(decision_tree_depth1)
    y_pred_depth1 = [predict_example(x, decision_tree_depth1) for x in X_test]
    conf_matrix_depth1 = confusion_matrix(y_test, y_pred_depth1)
    print("Confusion Matrix for depth=1:")
    print(conf_matrix_depth1)
    decision_tree_depth2 = id3(X_train, y_train, max_depth=2)
    visualize(decision_tree_depth2)
    y_pred_depth2 = [predict_example(x, decision_tree_depth2) for x in X_test]
    conf_matrix_depth2 = confusion_matrix(y_test, y_pred_depth2)
    print("Confusion Matrix for depth=2:")
    print(conf_matrix_depth2)

    # scikit-learn for monks-1
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    dot_data = export_graphviz(clf, out_file=None, feature_names=[f'x{i}' for i in range(1, 7)], class_names=['0', '1'], filled=True)
    graph = graphviz.Source(dot_data)
    graph.render(f'monks-1_sklearn_tree')
    graph.view()
    y_pred_sklearn = clf.predict(X_test)
    conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    print("Confusion Matrix for scikit-learn tree:")
    print(conf_matrix_sklearn)

    # Iris dataset example
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_binary = (X > X.mean(axis=0)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.3, random_state=42)
    clf_iris = DecisionTreeClassifier()
    clf_iris.fit(X_train, y_train)
    dot_data_iris = export_graphviz(clf_iris, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    graph_iris = graphviz.Source(dot_data_iris)
    graph_iris.render(f'iris_sklearn_tree')
    graph_iris.view()
    y_pred_iris = clf_iris.predict(X_test)
    conf_matrix_iris = confusion_matrix(y_test, y_pred_iris)
    print("Confusion Matrix for Iris dataset:")
    print(conf_matrix_iris)
