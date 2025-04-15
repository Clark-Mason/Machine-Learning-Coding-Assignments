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
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import zero_one_loss

#Import Sklearn Libraries for comparison


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    unique_num_features = np.unique(x)  # Number of unique features
    partitions = {v: np.where(x == v)[0] for v in unique_num_features}  # Dictionary to store partitions for each feature value pair
    return partitions


def entropy(y, weights=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z. 
    Include the weights of the boosted examples if present

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    if weights is None:
        weights = np.ones(len(y))/ len(y)  # If no weights are provided, assume equal weights
    
    ent = 0.0
    unique_values = np.unique(y)
    for v in unique_values:
        idx = np.where(y == v)[0]  # Get indices of the unique value
        p = np.sum(weights[idx]) / np.sum(weights)  # Probability of the unique value
        if p > 0:  # Avoid log(0)
            ent -= p * np.log2(p)  # Entropy calculation
    return  ent

    raise Exception('Function not yet implemented!')


def mutual_information(x, y, weights=None):
    """
    
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)

    Compute the weighted mutual information for Boosted learners
    """

    # INSERT YOUR CODE HERE
    entropy_y = entropy(y, weights)
    partitions = partition(x)
    conditional_entropy = 0.0
    total_weight = np.sum(weights) if weights is not None else len(y)
    for v, indices in partitions.items():
        if weights is None:
            weight_ratio = len(indices) / len(y)
        else:
            weight_ratio = np.sum(weights[indices]) / total_weight
        conditional_entropy += weight_ratio * entropy(y[indices], weights[indices] if weights is not None else None)
    return entropy_y - conditional_entropy


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
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
        info_gain = mutual_information(condition, y, weights)
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
        tree[(feature_index, candidate_value, True)] = id3(x[true_indices], y[true_indices], remaining_pairs, depth+1, max_depth,
                                                weights[true_indices] if weights is not None else None)
    if len(false_indices) == 0:
        tree[(feature_index, candidate_value, False)] = np.bincount(y).argmax()
    else:
        tree[(feature_index, candidate_value, False)] = id3(x[false_indices], y[false_indices], remaining_pairs, depth+1, max_depth,
                                                 weights[false_indices] if weights is not None else None)
    return tree


def bootstrap_sampler(x, y, num_samples):
    sample_mean = []
    for i in range (num_samples):
        sample = np.random.choice(len(x), size=len(x), replace=True)
        sample_mean.append((x[sample], y[sample]))
    return sample_mean


def bagging(x, y, max_depth, num_trees):
    """
    Implements bagging of multiple id3 trees where each tree trains on a boostrap sample of the original dataset
    """
    
    bootstrap_samples = bootstrap_sampler(x, y, num_trees)
    ensemble = []
    for x_sample, y_sample in bootstrap_samples:
        tree = id3(x_sample, y_sample, max_depth=max_depth)
        ensemble.append((1.0, tree))
    return ensemble
    raise Exception('Bagging not yet implemented!')


def boosting(x, y, max_depth, num_stumps):

    """
    Implements an adaboost algorithm using the id3 algorithm as a base decision tree
    """
    n = x.shape[0]
    weights = np.ones(n) / n
    ensemble = []  # Store (alpha_t, h_t) pairs

    y_mapped = np.where(y == 1, -1, 1)  # Map labels to -1 and 1

    for t in range(num_stumps):
        stump = id3(x, y, max_depth=max_depth, weights=weights)
        
        y_pred = np.array([predict_example_ens(x_i, [(1.0, stump)]) for x_i in x])
        y_pred_mapped = np.where(y_pred == 1, 1, -1)

        incorrect = (y_pred != y).astype(int)
        error = np.sum(weights * incorrect)

        error = np.clip(error, 1e-10, 1 - 1e-10)

        alpha = 0.5 * np.log((1 - error) / error)

        weights *= np.exp(-alpha * y_mapped * y_pred_mapped)

        weights /= np.sum(weights)

        ensemble.append((alpha, stump))

    return ensemble


def predict_example_ens(x, h_ens):
    """
    Predicts the classification label for a single example x using a combination of weighted trees
    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    total_vote = 0
    for alpha, tree in h_ens:
        node = tree
        # Traverse the tree manually
        while isinstance(node, dict):
            for (attr_index, attr_value, is_equal), subtree in node.items():
                if (x[attr_index] == attr_value) == is_equal:
                    node = subtree
                    break
        # Add weighted vote from final prediction
        vote = 1 if node == 1 else -1
        total_vote += alpha * vote
    return 1 if total_vote >= 0 else 0


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    return np.mean(y_true != y_pred)
    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    #BAGGING
    for max_depth in [3, 5]:
        for num_trees in [10, 20]:
            # Train ensemble using bagging
            ensemble = bagging(Xtrn, ytrn, max_depth=max_depth, num_trees=num_trees)
            # Obtain predictions by aggregating weighted votes over the ensemble
            predictions = [predict_example_ens(x, ensemble) for x in Xtst]
            cm = confusion_matrix(ytst, predictions)
            print("Bagging for max_depth = {}, num_trees = {}:".format(max_depth, num_trees))
            print(cm)
            print()

    #BOOSTING
    for max_depth in [1, 2]:
        for num_stumps in [20, 40]:
            # Train ensemble using boosting
            ensemble = boosting(Xtrn, ytrn, max_depth=max_depth, num_stumps=num_stumps)
            # Obtain predictions from the boosted ensemble
            predictions = [predict_example_ens(x, ensemble) for x in Xtst]
            cm = confusion_matrix(ytst, predictions)
            error_rate = zero_one_loss(ytst, predictions)
            print(error_rate*100)
            print("Boosting for max_depth = {}, num_stumps = {}:".format(max_depth, num_stumps))
            print(cm)
            print()

    #BAGGING via SCIKIT-LEARN
    for max_depth in [3, 5]:
        for n_estimators in [10, 20]:
            base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            clf = BaggingClassifier(estimator=base_estimator, n_estimators=n_estimators,
                                    bootstrap=True, random_state=42)
            clf.fit(Xtrn, ytrn)
            predictions = clf.predict(Xtst)
            cm = confusion_matrix(ytst, predictions)
            print("scikit-learn Bagging for max_depth = {}, n_estimators = {}:".format(max_depth, n_estimators))
            print(cm)
            print()
    
    #BOOSTING via SCIKIT-LEARN
    for max_depth in [1, 2]:
        for n_estimators in [20, 40]:
            base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=42)
            clf.fit(Xtrn, ytrn)
            predictions = clf.predict(Xtst)
            cm = confusion_matrix(ytst, predictions)
            print("scikit-learn AdaBoost for max_depth = {}, n_estimators = {}:".format(max_depth, n_estimators))
            print(cm)
            print()
