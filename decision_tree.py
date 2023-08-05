"""This module includes methods for training and predicting using decision trees."""
import numpy as np


def calculate_information_gain(data, labels):
    """
    Computes the information gain for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape

    parent_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            parent_entropy -= class_prob * np.log(class_prob)

    # print("Parent entropy is %d\n" % parent_entropy)

    # initialization of gains for every attribute
    gain = parent_entropy * np.ones(d)

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n))  # number of examples containing each feature
    prob_x = num_x / n  # fraction of examples containing each feature
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            # probability of observing class c for each feature
            prob_y_given_x = num_y_and_x / (num_x + 1e-8)
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - \
                    np.multiply(np.multiply(prob_x, prob_y_given_x),
                                np.log(prob_y_given_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - \
                    np.multiply(np.multiply(
                        prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    return recursive_tree_train(
        train_data, train_labels, depth=0, max_depth=params['max_depth'])


def recursive_tree_train(data, labels, depth, max_depth):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 0:
        node = {"is_leaf": True,
                "prediction": 0}
        return node
    elif unique_labels.size == 1 or depth == max_depth:
        node = {"is_leaf": True,
                "prediction": unique_labels[np.argmax(label_counts)]}
        return node
    else:
        gain = calculate_information_gain(data, labels)
        best_feature = gain.argmax()
        feature_values = data[best_feature, :]
        left_mask = feature_values == 0
        right_mask = feature_values == 1
        left_data, left_labels = data[:, left_mask], labels[left_mask]
        right_data, right_labels = data[:, right_mask], labels[right_mask]
        return {"is_leaf": False,
                "splitting_feature": best_feature,
                "left": recursive_tree_train(
                    left_data, left_labels, depth + 1, max_depth),
                "right": recursive_tree_train(
                    right_data, right_labels, depth + 1, max_depth)}


def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    n = data.shape[1]
    predicted_labels = np.zeros(n)
    for i in range(n):
        node = model
        while not node["is_leaf"]:
            if data[node["splitting_feature"], i] == 0:
                node = node["left"]
            else:
                node = node["right"]
        predicted_labels[i] = node["prediction"]
    return predicted_labels
