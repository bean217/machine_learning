import numpy as np
import heapq


from sim_measure import manhattan, euclidean, chebyshev, cosine


def select_sim_measure(sim_measure):
    sm = None
    name = "custom"
    if sim_measure in ["L1", "manhattan"]:
        sm = manhattan
        name = "manhattan"
    elif sim_measure in ["L2", "euclidean"]:
        sm = euclidean
        name = "euclidean"
    elif sim_measure in ["max", "chebyshev"]:
        sm = chebyshev
        name = "chebyshev"
    elif sim_measure == "cosine":
        sm = cosine
        name = "cosine"
    else:
        if callable(sim_measure) and sim_measure.__code__.co_argcount == 2:
            sm = sim_measure
            name = "custom"
        else:
            raise Exception("similarity measure doesn't exist")
    # return sim measure func and name
    return (sm, name)

class KNN:
    """Simple implementation of a K-Nearest-Neighbors classifier that is
        capable of evaluating a validation set and generating a report.
    """
    def __init__(self, num_neighbors = 1, sim_measure = "L2"):
        """KNN constructor
        @param num_neighbors: number of neighbors to consider (default = 1)
        @param sim_measure: name of similarity measure to use (default = "L2").
            A custom similarity measure can also be used here.
        """
        # based on sim_measure parameter, determine distance function to use
        sm, name = select_sim_measure(sim_measure)
        # similiarity measure function
        self.__sm = sm
        # number of neighbors to consider
        self.__n = num_neighbors
        # training data
        self.__training_data = None
        # training labels
        self.__training_labels = None
        # name of similarity measure
        self.__sm_name = name
        # most recent evaluation measures
        self.eval_info = None
    

    def set_num_neighbors(self, new_n):
        self.__n = new_n

    
    def set_sim_measure(self, new_sim = "L2"):
        sm, name = select_sim_measure(new_sim)
        self.__sm = sm
        self.__sm_name = name


    def fit(self, data, labels):
        """Fits training data to the model
        @param data: array of training data inputs
        @param labels: array of training data labels
        """
        self.__training_data = np.asarray(data)
        self.__training_labels = np.asarray(labels)
        # amount of training data and labels must be equal
        assert self.__training_data.shape[0] == self.__training_labels.shape[0]
            

    def classify(self, input):
        """Classifies an input based on the current fit training data and
            and classification parameters
            @param input: unobserved training instance being classified
            @returns: classification of the unobserved input
        """
        d = np.asanyarray(input)
        # make sure input is the same dimensionality as the training data
        assert d.shape == self.__training_data.shape[1:]

        # flatten matrix data for use in similarity measures
        d = d.flatten()

        # for each training example, calculate the distance to the input
        distances = [self.__sm(input, np.asarray(e).flatten()) for e in self.__training_data]
        smallest_indices = heapq.nsmallest(
            self.__n, range(len(distances)), key=distances.__getitem__)
        
        # classify as the most common result
        if self.__n == 1:
            # if there is only one neighbor, classify as its label
            return self.__training_labels[smallest_indices[0]]
        else:
            # get the classifications associated with the smallest distances
            nn_classes = [self.__training_labels[i] for i in smallest_indices]
            # return the most common classification
            return max(set(nn_classes), key = nn_classes.count)


    def evaluate(self, v_data, v_labels):
        """Attempts to classify a set of unobserved validations examples, the results
            of which are stored in self.eval_info.
            @param v_data: input data of validation set
            @param v_labels: labels of validation set
            @returns: accuracy
        """
        test_data = np.asarray(v_data)
        test_labels = np.asarray(v_labels)
        # amount of training data and labels must be equal
        assert test_data.shape[0] == test_labels.shape[0]
        # input data and labels have the same dimensionality
        assert test_data.shape[1:] == self.__training_data.shape[1:]
        assert test_labels.shape[1:] == self.__training_labels.shape[1:]

        # perform evaluation
        correct = 0
        incorrect = 0
        for i, val in enumerate(v_data):
            c = self.classify(val)
            
            if c == v_labels[i]:
                correct += 1
            else:
                incorrect += 1

        # set last evaluation data
        self.eval_info = {
            "neighbors": self.__n,
            "sim_measure_name": self.__sm_name,
            "sim_measure": self.__sm,
            "correct": correct,
            "incorrect": incorrect}
        
        # return accuracy
        return correct / (correct + incorrect)


    def eval_report(self):
        """Displays the evaludation report based on the most recent call to self.evaluate.
        """
        assert self.eval_info is not None
        print("=============== EVAL REPORT ===============\n" + \
            f'  neighbors:\t\t\t{self.eval_info["neighbors"]}\n' + \
            f'  simularity measure:\t\t{self.eval_info["sim_measure_name"]}\n' + \
            f'  correctly classified:\t\t{self.eval_info["correct"]}\n' + \
            f'  incorrectly classified:\t{self.eval_info["incorrect"]}\n' + \
            f'  total:\t\t\t{self.eval_info["correct"] + self.eval_info["incorrect"]}\n' + \
            f'  precision:\t\t\t{self.eval_info["correct"] / (self.eval_info["correct"] + self.eval_info["incorrect"])}\n' + \
            "===========================================\n")


