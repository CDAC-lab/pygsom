import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math

data_filename = "example/data/zoo.txt".replace('\\', '/')


class GSOM:

    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8,
                 max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000):
        """
        GSOM structure:
        keep dictionary to x,y coordinates and numpy array to keep weights
        :param spred_factor: spread factor of GSOM graph
        :param dimensions: weight vector dimensions
        :param distance: distance method: support scipy.spatial.distance.cdist
        :param initialize: weight vector initialize method
        :param learning_rate: initial training learning rate of weights
        :param smooth_learning_factor: smooth learning factor to change the initial smooth learning rate from training
        :param max_radius: maximum neighbourhood radius
        :param FD: spread weight value #TODO: check this from paper
        :param r: learning rate update value #TODO: check this from paper
        :param alpha: learning rate update value #TODO: check this from paper
        :param initial_node_size: initial node allocation in memory
        """
        self.initial_node_size = initial_node_size
        self.node_count = 0  # Keep current GSOM node count
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions))  # initialize node allocation in memory
        self.node_coordinate = np.zeros((self.initial_node_size, 2))  # initialize node coordinate in memory
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)  # initialize node error in memory
        self.spred_factor = spred_factor
        self.groth_threshold = -dimensions * math.log(self.spred_factor)
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.dimentions = dimensions
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        self.initialize_GSOM()
        self.node_labels = None  # Keep the prediction GSOM nodes
        self.output = None # keep the cluster id of each data point
        # HTM sequence learning parameters
        self.predictive = None  # Keep the prediction of the next sequence value (HTM predictive state)
        self.active = None  # Keep the activation of the current sequence value (HTM active state)
        self.sequence_weights = None  # Sequence weight matrix. This has the dimensions node count*column height


    def initialize_GSOM(self):
        self.insert_node_with_weights(1, 1)
        self.insert_node_with_weights(1, 0)
        self.insert_node_with_weights(0, 1)
        self.insert_node_with_weights(0, 0)

    def insert_new_node(self, x, y, weights):
        if self.node_count > self.initial_node_size:
            print("node size out of bound")
            # TODO:resize the nodes
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights
        self.node_coordinate[self.node_count][0] = x
        self.node_coordinate[self.node_count][1] = y
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            node_weights = np.random.rand(self.dimentions)
        else:
            print("initialize method not support")
            # TODO:: add other initialize methods
        self.insert_new_node(x, y, node_weights)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (self.node_list[self.map[(winnerx, winnery)]] + self.node_list[
            self.map[(next_nodex, next_nodey)]]) * 0.5
        return weights

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (2 * self.node_list[self.map[(winnerx, winnery)]] - self.node_list[
            self.map[(next_nodex, next_nodey)]])
        return weights

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        weights = np.full(self.dimentions, (max(self.node_list[self.map[(winnerx, winnery)]]) + min(
            self.node_list[self.map[(winnerx, winnery)]])) / 2)
        return weights

    def grow_node(self, wx, wy, x, y, side):
        """
        grow new node if not exist on x,y coordinates using the winner node weight(wx,wy)
        check the side of the winner new node add in following order (left, right, top and bottom)
        new node N
        winner node W
        Other nodes O
        left
        =============
        1 O-N-W
        -------------
        2 N-W-O
        -------------
        3   O
            |
          N-W
        -------------
        4 N-W
            |
            O
        -------------
        =============
        right
        =============
        1 W-N-O
        -------------
        2 o-W-N
        -------------
        3 O
          |
          W-N
        -------------
        4 W-N
          |
          O
        -------------
        =============
        top
        ===============
        1 O
          |
          N
          |
          W
        -------------
        1 N
          |
          W
          |
          O
        -------------
        3 N
          |
          W-N
        -------------
        4 N
          |
        O-N
        -------------
        =============
        :param wx:
        :param wy:
        :param x:
        :param y:
        :param side:
        """
        if not (x, y) in self.map:
            if side == 0:  # add new node to left of winner
                if (x - 1, y) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x - 1, y)
                elif (wx + 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
                elif (wx, wy + 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
                elif (wx, wy - 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 1:  # add new node to right of winner
                if (x + 1, y) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x + 1, y)
                elif (wx - 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
                elif (wx, wy + 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
                elif (wx, wy - 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 2:  # add new node to top of winner
                if (x, y + 1) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y + 1)
                elif (wx, wy - 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
                elif (wx + 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
                elif (wx - 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 3:  # add new node to bottom of winner
                if (x, y - 1) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y - 1)
                elif (wx, wy + 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
                elif (wx + 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
                elif (wx - 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            # clip the wight between (0,1)
            weights[weights < 0] = 0.0
            weights[weights > 1] = 1.0
            self.insert_new_node(x, y, weights)

    def spread_wights(self, x, y):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        self.node_errors[self.map[(x, y)]] = self.groth_threshold/2   #TODO check this value if different in Rashmika's
                                                                    # version and paper version (paper t/2 Rashmika t)
        self.node_errors[self.map[(leftx, lefty)]] *= (1 + self.FD)
        self.node_errors[self.map[(rightx, righty)]] *= (1 + self.FD)
        self.node_errors[self.map[(topx, topy)]] *= (1 + self.FD)
        self.node_errors[self.map[(bottomx, bottomy)]] *= (1 + self.FD)

    def adjust_wights(self, x, y, rmu_index):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        # Check all neighbours exist and spread the weights
        if (leftx, lefty) in self.map \
                and (rightx, righty) in self.map \
                and (topx, topy) in self.map \
                and (bottomx, bottomy) in self.map:
            self.spread_wights(x, y)
        else:
        # Grow new nodes for the four sides
            self.grow_node(x, y, leftx, lefty, 0)
            self.grow_node(x, y, rightx, righty, 1)
            self.grow_node(x, y, topx, topy, 2)
            self.grow_node(x, y, bottomx, bottomy, 3)
        self.node_errors[rmu_index] = self.groth_threshold/2 #TODO check the need of setting the error to zero after weight adaptation

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index, :].reshape(1, self.dimentions), self.distance)
        rmu_index = out.argmin()  # get winner node index
        error_val = out.min()
        # get winner node coordinates
        rmu_x = int(self.node_coordinate[rmu_index][0])
        rmu_y = int(self.node_coordinate[rmu_index][1])

        # Update winner error
        error = data[data_index] - self.node_list[rmu_index]
        self.node_list[self.map[(rmu_x, rmu_y)]] = self.node_list[self.map[(rmu_x, rmu_y)]] + learning_rate * error

        # Get integer radius value
        mask_size = round(radius)

        # Iterate over the winner node radius(neighbourhood)
        for i in range(rmu_x - mask_size, rmu_x + mask_size):
            for j in range(rmu_y - mask_size, rmu_y + mask_size):
                # Check neighbour coordinate in the map not winner coordinates
                if (i, j) in self.map and (i != rmu_x and j != rmu_y):
                    # get error between winner and neighbour
                    error = self.node_list[rmu_index] - self.node_list[self.map[(i, j)]]
                    distance = (rmu_x - i) * (rmu_x - i) + (rmu_y - j) * (rmu_y - j)
                    eDistance = np.exp(-1.0 * distance / (2.0 * (radius * radius)))  # influence from distance

                    # Update neighbour error
                    self.node_list[self.map[(i, j)]] = self.node_list[self.map[(i, j)]] \
                                                       + learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        # Iterate all data points
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        # Iterate all data points
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

            # winner node weight update and grow
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.groth_threshold:
                self.adjust_wights(rmu_x, rmu_y, rmu_index)

    def fit(self, data, training_iterations, smooth_iterations):
        """
        method to train the GSOM map
        :param data:
        :param training_iterations:
        :param smooth_iterations:
        """
        current_learning_rate = self.learning_rate
        # growing iterations
        for i in tqdm(range(training_iterations)):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.grow(data, radius_exp, current_learning_rate)

        # smoothing iterations
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations)):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.smooth(data, radius_exp, current_learning_rate)
        # Identify winners
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data, self.distance)
        return out.argmin(axis=0)

    def predict(self, data, index_col, label_col=None):
        """
        Identify the winner nodes for test dataset
        Predict the winning node for each data point and create a pandas dataframe
        need to provide both index column and label column
        :param data:
        :param index_col:
        :param label_col:
        :return:
        """

        # Prepare the dataset - remove label and index column
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)

        weight_columns.remove(index_col)
        data_n = data[weight_columns].to_numpy()
        data_out = pd.DataFrame(data[output_columns])
        # Identify winners
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        data_out["output"] = out.argmin(axis=0)

        grp_output =data_out.groupby("output")
        dn = grp_output[index_col].apply(list).reset_index()
        dn = dn.set_index("output")
        if label_col:
            dn[label_col] = grp_output[label_col].apply(list)
        dn = dn.reset_index()
        dn["hit_count"] = dn[index_col].apply(lambda x: len(x))
        dn["x"] = dn["output"].apply(lambda x: self.node_coordinate[x, 0])
        dn["y"] = dn["output"].apply(lambda x: self.node_coordinate[x, 1])
        hit_max_count = dn["hit_count"].max()
        self.node_labels = dn
        # display map
        #plot(self.node_labels, index_col)
        self.output = data_out

        return self.node_labels


if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv(data_filename)
    print(df.shape)
    data_training = df.iloc[:, 1:17]
    gsom = GSOM(.83, 16, max_radius=4)
    gsom.fit(data_training.to_numpy(), 100, 50)
    output = gsom.predict(df,"Name","label")
    output.to_csv("output.csv", index=False)

    print("complete")
