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


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector x.
    """
    unique_values = np.unique(x)
    partitions = {v: np.where(x == v)[0] for v in unique_values}
    return partitions


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk) in y

    Returns the entropy of y: H(y) = - [ p(y=v1) log2(p(y=v1)) + ... + p(y=vk) log2(p(y=vk)) ]
    """
    unique_values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). 
    The data column is a binary attribute (n x 1) indicating whether a given candidate split is true.
    
    Mutual information is defined as:
        I(x, y) = H(y) - H(y | x)
    where H(y | x) is the weighted average entropy after the split.
    """
    entropy_y = entropy(y)
    partitions = partition(x)
    conditional_entropy = 0.0

    for value, indices in partitions.items():
        y_partition = y[indices]
        conditional_entropy += (len(y_partition) / len(y)) * entropy(y_partition)

    return entropy_y - conditional_entropy


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions:
        1. If the entire set of labels (y) is pure (all y are the same), then return that label.
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label).
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label).
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion,
    and partitions the data set into two parts based on the binary question "is feature i equal to value v?".
    
    The tree is stored as a nested dictionary, where each entry is of the form:
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion.
    * The subtree itself can be a nested dictionary or a single label (leaf node).
    * Leaf nodes are (majority) class labels.

    Returns a decision tree represented as a nested dictionary.
    """
    # Create the attribute-value pairs if not provided.
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(x.shape[1]):
            unique_values = np.unique(x[:, i])
            for value in unique_values:
                attribute_value_pairs.append((i, value))

    # Termination Condition 1: Pure labels.
    unique_labels = np.unique(y)
    if len(unique_labels) == 1:
        return unique_labels[0]
    
    # Termination Condition 2 and 3: No features left or maximum depth reached.
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return np.bincount(y).argmax()

    # Select the best attribute-value pair by evaluating mutual information on the binary test.
    max_info_gain = -1
    best_pair = None
    for pair in attribute_value_pairs:
        # Create a binary feature based on the candidate split: True if feature equals candidate value.
        binary_feature = (x[:, pair[0]] == pair[1])
        info_gain = mutual_information(binary_feature, y)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_pair = pair

    if max_info_gain == 0:
        return np.bincount(y).argmax()

    # Remove the chosen candidate from the list.
    remaining_pairs = [pair for pair in attribute_value_pairs if pair != best_pair]

    # Partition the data into two subsets using the binary test.
    tree = {}
    binary_test = (x[:, best_pair[0]] == best_pair[1])
    
    # True branch: where the test is True.
    x_true = x[binary_test]
    y_true = y[binary_test]
    if len(y_true) == 0:
        tree[(best_pair[0], best_pair[1], True)] = np.bincount(y).argmax()
    else:
        tree[(best_pair[0], best_pair[1], True)] = id3(x_true, y_true, remaining_pairs, depth + 1, max_depth)
    
    # False branch: where the test is False.
    x_false = x[~binary_test]
    y_false = y[~binary_test]
    if len(y_false) == 0:
        tree[(best_pair[0], best_pair[1], False)] = np.bincount(y).argmax()
    else:
        tree[(best_pair[0], best_pair[1], False)] = id3(x_false, y_false, remaining_pairs, depth + 1, max_depth)
    
    return tree


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label (leaf node) is reached.

    Returns the predicted label of x according to tree.
    """
    for key, subtree in tree.items():
        attribute_index, attribute_value, is_equal = key
        # Check if the example meets the binary condition for this node.
        if (x[attribute_index] == attribute_value) == is_equal:
            if isinstance(subtree, dict):
                return predict_example(x, subtree)
            else:
                return subtree
    return None


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred).

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    return np.mean(y_true != y_pred)


def visualize(tree, depth=0):
    """
    Pretty prints the decision tree to the console, showing whether a branch
    corresponds to the True (==) or False (!=) condition.
    """
    if depth == 0:
        print('TREE')

    for key, subtree in tree.items():
        attribute_index, attribute_value, is_equal = key
        condition_str = f"x{attribute_index} == {attribute_value}" if is_equal else f"x{attribute_index} != {attribute_value}"
        print('|\t' * depth, end='')
        print('+-- [SPLIT: {0}]'.format(condition_str))
        if isinstance(subtree, dict):
            visualize(subtree, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(subtree))



if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
