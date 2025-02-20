Here’s a concise summary of all the functions in the `decision_tree.py` file, their purposes, and how they contribute to the decision tree construction, formatted as a `README.md` file:

---

# Decision Tree Implementation (ID3 Algorithm)

This Python script implements the ID3 algorithm for building a decision tree. Below is a summary of the functions and their roles in constructing the decision tree.

---

## Functions

### 1. **`partition(x)`**
- **Purpose**: Partitions a column vector `x` into subsets based on its unique values.
- **Contribution**: Used to split the dataset into subsets for evaluating potential splits during tree construction.
- **Output**: A dictionary where keys are unique values in `x`, and values are indices of rows where `x` equals the key.

---

### 2. **`entropy(y)`**
- **Purpose**: Computes the entropy of a vector `y`, which measures the impurity or uncertainty in the labels.
- **Contribution**: Helps determine the purity of a dataset. Lower entropy indicates a more homogeneous dataset, which is desirable for splitting.
- **Formula**:
  \[
  H(y) = -\sum_{i=1}^{k} p(y=v_i) \cdot \log_2(p(y=v_i))
  \]

---

### 3. **`mutual_information(x, y)`**
- **Purpose**: Computes the mutual information (information gain) between a binary feature `x` and the labels `y`.
- **Contribution**: Used as the splitting criterion to select the best attribute-value pair for splitting the dataset.
- **Formula**:
  \[
  I(x, y) = H(y) - H(y | x)
  \]
  where \( H(y | x) \) is the conditional entropy of `y` given `x`.

---

### 4. **`id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5)`**
- **Purpose**: Implements the ID3 algorithm to recursively build a decision tree.
- **Contribution**: Constructs the tree by:
  1. Checking termination conditions (pure labels, no attributes left, or maximum depth reached).
  2. Selecting the best attribute-value pair using mutual information.
  3. Partitioning the data and recursively building subtrees.
- **Output**: A decision tree represented as a nested dictionary.

---

### 5. **`predict_example(x, tree)`**
- **Purpose**: Predicts the label for a single example `x` by traversing the decision tree.
- **Contribution**: Used to evaluate the performance of the decision tree by predicting labels for test data.
- **How It Works**: Recursively traverses the tree, checking conditions at each node until a leaf node (predicted label) is reached.

---

### 6. **`compute_error(y_true, y_pred)`**
- **Purpose**: Computes the average error between the true labels (`y_true`) and the predicted labels (`y_pred`).
- **Contribution**: Evaluates the accuracy of the decision tree.
- **Formula**:
  \[
  \text{Error} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}}^{(i)} \neq y_{\text{pred}}^{(i)})
  \]

---

### 7. **`visualize(tree, depth=0)`**
- **Purpose**: Pretty-prints the decision tree to the console, showing splits and leaf nodes.
- **Contribution**: Helps visualize and debug the tree structure.
- **How It Works**: Recursively traverses the tree and prints the conditions and labels at each level.

---

## Workflow
1. **Data Preparation**: Load the dataset into feature matrix `x` and label vector `y`.
2. **Tree Construction**: Call `id3` to build the decision tree.
3. **Visualization**: Use `visualize` to print the tree structure.
4. **Prediction**: Use `predict_example` to predict labels for test data.
5. **Evaluation**: Use `compute_error` to calculate the error rate.

---

## Example
```python
# Load data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Build tree
decision_tree = id3(x, y, max_depth=2)

# Visualize tree
visualize(decision_tree)

# Predict and evaluate
y_pred = [predict_example(x_i, decision_tree) for x_i in x]
error = compute_error(y, y_pred)
print(f'Error: {error * 100:.2f}%')
```

---

## Key Concepts
- **Entropy**: Measures dataset impurity.
- **Mutual Information**: Evaluates the quality of a split.
- **Recursive Splitting**: The ID3 algorithm recursively splits the dataset based on the best attribute-value pair.
- **Termination Conditions**: Stop splitting if labels are pure, no attributes are left, or maximum depth is reached.

---

This implementation is a classic example of the ID3 algorithm, focusing on information gain as the splitting criterion. It avoids using external machine learning libraries, making it a great learning tool for understanding decision trees.
