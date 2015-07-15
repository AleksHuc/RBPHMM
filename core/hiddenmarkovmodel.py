from discretestate import DiscreteState
from discreteemission import DiscreteEmission

import numpy
import math
import copy
import pickle


class HiddenMarkovModel:
    """
    Class that represent hidden Markov model.

    :param emission_alphabet: List of strings that represent emission symbols.
    :param state_alphabet: List of strings that represent state symbols.
    :param state_type: Single string that represents the type of states.

    Specification of an HMM:
        1. N, the number if states in the model. We denote the set of all possible states as S = {S_{1}, S_{2}, ... , S_{N}}, the state at time t as q_{t}.
        2. M, the number of distinct observation symbols per state, i.e., the discrete alphabet size of the output set, We denote the set of all possible output symbols as V = {v_{1}, v_{2}, ... , v_{m}},
            the output symbol at the time t is O_{t}. The sequence of observed symbols is denoted as O = O_{1} * O_{2} * ... * O_{T}.
        3. The state transition probability distribution A = {a_{ij}}, where a_{ij} = P[q_{t+1} = S_{j} | q_{t} = S_{i}], 1 <= i, j <= N.
        4. The observation symbol probability distribution in state j, B = {b_{j}(k)}, where b_{j}(k) = P[O_{t} = v_{k} | q_{t} = S_{j}], 1 <= j <= N, 1 <= k <= M.
        5. The initial state distributions Pi = {Pi_{i}}, where Pi_{i} = P[q_{1} = S_{i}], 1 <= i <= N.
    """

    def __init__(self, emission_alphabet, state_alphabet, state_type):
        if state_type == "discrete":
            self.emission_alphabet = self.make_emission_alphabet(emission_alphabet, state_type)
            self.state_alphabet = self.make_emission_alphabet(state_alphabet, state_type)
            self.emission_matrix = self.make_emission_matrix(self.state_alphabet, self.emission_alphabet, 1.0 / len(self.emission_alphabet))
            self.transition_matrix = self.make_transition_matrix(self.state_alphabet, 1.0 / len(self.state_alphabet))
            self.number_of_emissions = 0
        else:
            print("ERROR: State type is wrong!")

    @staticmethod
    def make_emission_alphabet(emission_alphabet, state_type):
        """
        Method that wraps list of emission symbols in tuples of Emission object and emission index.

        :param emission_alphabet: List that represent emission symbols.
        :param state_type: Single string that represents the type of states.
        :returns:  List of tuples of Emission object and emission index.
        """

        emission_alphabet_list = []
        index = 0
        if state_type == "discrete":
            for i in emission_alphabet:
                emission_alphabet_list.append((DiscreteEmission(i), index))
                index += 1
        else:
            print("ERROR: State type is wrong!")

        return emission_alphabet_list

    @staticmethod
    def make_state_alphabet(state_alphabet, state_type):
        """
        Method that wraps list of state symbols in tuples of State object and state index.

        :param state_alphabet: List that represent state symbols.
        :param state_type: Single string that represents the type of states.
        :returns:  List of tuples of State object and state index.
        """

        state_alphabet_list = []
        index = 0
        if state_type == "discrete":
            for i in state_alphabet:
                state_alphabet_list.append((DiscreteState(i), index))
                index += 1
        else:
            print("ERROR: State type is wrong!")

        return state_alphabet_list

    @staticmethod
    def make_transition_matrix(state_alphabet, initial_value):
        """
        Method that makes transition matrix for given state alphabet.

        :param state_alphabet: List that represent state symbols.
        :param initial_value: Number that denotes the inital values in matrix.
        :returns: Matrix with initial transition probabilities between all states.
        """

        transition_matrix = {}
        for i in state_alphabet:
            state_matrix = {}
            for j in state_alphabet:
                state_matrix[j] = initial_value
            transition_matrix[i] = state_matrix

        return transition_matrix

    @staticmethod
    def make_emission_matrix(state_alphabet, emission_alphabet, initial_value):
        """
        Method that makes emission matrix for given state and emission alphabet.

        :param state_alphabet: List that represent state symbols.
        :param emission_alphabet: List that represent emission symbols.
        :param initial_value: Number that denotes the inital values in matrix.
        :returns: Matrix with initial emission probabilities of all emissions for each state.
        """

        state_matrix = {}
        for i in state_alphabet:
            emission_matrix = {}
            for j in emission_alphabet:
                emission_matrix[j] = initial_value
            state_matrix[i] = emission_matrix

        return state_matrix

    @staticmethod
    def print_matrix(current_matrix):
        """
        Method that prints to console the given matrix.

        :param current_matrix: Given matrix.
        """
        print [(x[0][0].name, x[1]) for x in sorted(current_matrix[current_matrix.keys()[1]].items(), key=lambda x: x[1], reverse=True)]
        #print current_matrix[current_matrix.keys()[1]]

        print sorted([k[0].name for k in current_matrix[current_matrix.keys()[0]]])

        first_line = '{:>20}'.format("|") + "".join(['{:>20}'.format(i[0].name + "|") for i in current_matrix[current_matrix.keys()[0]]])
        print first_line
        print "".join(["-" for _ in xrange(len(first_line))])
        for i in current_matrix:
            line = '{:>20}'.format(i[0].name + "|")
            for j in current_matrix[i]:
                line += '{:>20}'.format(str(current_matrix[i][j]) + "|")
            print line
        print "".join(["-" for _ in xrange(len(first_line))])

    @staticmethod
    def matrix_log_conversion(current_matrix):
        """
        Method that converts probabilities of matrix to logarithmic values.

        :param current_matrix: Given matrix.
        :returns: Matrix with logarithmic values.
        """
        matrix = copy.deepcopy(current_matrix)
        for i in current_matrix:
            for j in current_matrix[i]:
                matrix[i][j] = math.log10(current_matrix[i][j])

        return matrix

    def matrix_eln_conversion(self, current_matrix):
        """
        Method that converts probabilities of matrix to logarithmic values.

        :param current_matrix: Given matrix.
        :returns: Matrix with logarithmic values.
        """
        matrix = copy.deepcopy(current_matrix)
        for i in current_matrix:
            for j in current_matrix[i]:
                matrix[i][j] = self.eln(current_matrix[i][j])

        return matrix

    @staticmethod
    def eexp(x):
        """
        The extended exponential function is the standard exponential function e^{x}, except that it is extended to handle log zero, and is defined as follows:

        1. for a real number x: eexp(x) = e^{x}
        2. for x = LOGZERO: eexp(x) = 0

        :param x: Input value.
        :returns: Exponential value of input value.
        """

        if x == "LOGZERO":
            return 0.0
        else:
            return math.exp(x)

    @staticmethod
    def eln(x):
        """
        The extended logarithm is the standard logarithm provided, except that it is extended to handle inputs of zero, and is defined as follows:

        1. for positive real numbers x: eln(x) = ln(x)
        2. for x = 0: eln(x) = LOGZERO

        :param x: Input value.
        :returns: Logarithmic value of input value.
        """

        if x == 0.0:
            return "LOGZERO"
        elif x > 0.0:
            return math.log(x)
        else:
            return "Negative input error."

    @staticmethod
    def elnproduct(x, y):
        """
        The extended logarithm product function returns the logarithm of the product of x and y, and is defined as follows:

        1. for positive real y and y: elnproduct(eln(x), eln(y)) = eln(x) + eln(y)
        2. for x = 0: elnproduct(LOGZERO, eln(y)) = LOGZERO
        3. for y = 0: elnproduct(eln(x), LOGZERO) = LOGZERO

        :param x: First logarithmic input value.
        :param y: Second logarithmic input value.
        :returns: The logarithmic product of input values.
        """

        if x == "LOGZERO" or y == "LOGZERO":
            return "LOGZERO"
        else:
            return x + y

    @staticmethod
    def elndivision(x, y):
        """
        The extended logarithm product function returns the logarithm of the division of x and y, and is defined as follows:

        1. for positive real y and y: elndivision(eln(x), eln(y)) = eln(x) - eln(y)
        2. for x = 0: elndivision(LOGZERO, eln(y)) = LOGZERO
        3. for y = 0: elndivision(eln(x), LOGZERO) = LOGZERO

        :param x: First logarithmic input value.
        :param y: Second logarithmic input value.
        :returns: The logarithmic product of input values.
        """

        if x == "LOGZERO" or y == "LOGZERO":
            return "LOGZERO"
        else:
            return x - y

    def elnsum(self, x, y):
        """
        The extended logarithm sum function computes the extended logarithm of the sum of x and y given as inputs the extended logarithm of x an y, and is defines as follows:

        1. for positive real x and y: elnsum(eln(x), eln(y)) = eln(x + y)
        2. for x = 0: elnsum(LOGZERO, eln(y)) = eln(y)
        3. for y = 0: elnsum(eln(x), LOGZERO) = eln(x)

        :param x: First logarithmic input value.
        :param y: Second logarithmic input value.
        :returns: The logarithmic sum of input values.
        """

        if x == "LOGZERO" or y == "LOGZERO":
            if x == "LOGZERO":
                return y
            else:
                return x
        else:
            if x > y:
                return x + self.eln(1 + math.exp(y - x))
            else:
                return y + self.eln(1 + math.exp(x - y))

    def create_state_matrix(self, sequence):
        """
        Method that reorganizes state sequence in state matrix where each row represents each state and 1 in each column represents which state is active at current location in sequence.

        :param sequence: Current sequence.
        :returns: Matrix that represents state sequence.
        """

        matrix = numpy.zeros((len(sequence[0]) + 1) * len(self.state_alphabet)).reshape(len(self.state_alphabet), len(sequence[0]) + 1)

        for i in xrange(len(sequence[0])):
            matrix[self.get_state_by_name(sequence[0][i])[1]][i + 1] = 1

        matrix[:, 0] = matrix[:, 1]

        return matrix

    def get_state_by_name(self, name):
        """
        Method that finds State index tuple in state alphabet.

        :param name: Name of State object.
        :returns: State index tuple for given name.
        """

        return [i for i in self.state_alphabet if i[0].name == name][0]

    def get_state_by_index(self, index):
        """
        Method that finds State index tuple in state alphabet.

        :param index: Index of tuple.
        :returns: State index tuple for given index.
        """

        return [i for i in self.state_alphabet if i[1] == index][0]

    def get_emission_by_name(self, name):
        """
        Method that finds Emission index tuple in emission alphabet.

        :param name: Name of Emission object.
        :returns: Emission index tuple for given name.
        """

        return [i for i in self.emission_alphabet if i[0].name == name][0]

    def get_emission_by_index(self, index):
        """
        Method that finds Emission index tuple in emission alphabet.

        :param index: Index of tuple.
        :returns: Emission index tuple for given index.
        """

        return [i for i in self.emission_alphabet if i[1] == index][0]

    def state_sequence_training(self, sequences):
        """
        Method that trains current hidden Markov model with given sequences of states and emissions, for combining probabilities we use weighted sum, where weights are represented by sequence lengths.

        :param sequences: List of string of states and string of emissions.
        :returns: Array of transition and emission matrices for each sequence and all other sequence data.
        """

        matrices = []
        number_of_emissions = 0
        entire_transition_matrix = self.make_transition_matrix(self.state_alphabet, 0)
        entire_emission_matrix = self.make_emission_matrix(self.state_alphabet, self.emission_alphabet, 0)

        index = 0
        for sequence in sequences:
            #print "Sequence: ", index
            index += 1

            transition_matrix_count = self.make_transition_matrix(self.state_alphabet, 0)
            transition_matrix = self.make_transition_matrix(self.state_alphabet, 0)
            emission_matrix_count = self.make_emission_matrix(self.state_alphabet, self.emission_alphabet, 0)
            emission_matrix = self.make_emission_matrix(self.state_alphabet, self.emission_alphabet, 0)

            if len(sequence[0]) == len(sequence[1]):

                number_of_emissions += len(sequence[0])
                for i in xrange(len(sequence[0])):

                    if i < len(sequence[0]) - 1:
                        transition_matrix_count[self.get_state_by_name(sequence[0][i])][self.get_state_by_name(sequence[0][i + 1])] += 1

                    emission_matrix_count[self.get_state_by_name(sequence[0][i])][self.get_emission_by_name(sequence[1][i])] += 1

                for i in transition_matrix:
                    number_of_transitions = sum(transition_matrix_count[i].itervalues())
                    for j in transition_matrix[i]:
                        if number_of_transitions == 0:
                            transition_matrix[i][j] = 0.0
                        else:
                            transition_matrix[i][j] = float(transition_matrix_count[i][j]) / float(number_of_transitions)
                        entire_transition_matrix[i][j] += transition_matrix[i][j] * len(sequence[0])

                for i in emission_matrix:
                    number_of_emission = sum(emission_matrix_count[i].itervalues())
                    for j in emission_matrix[i]:
                        if number_of_emission == 0:
                            emission_matrix[i][j] = 0.0
                        else:
                            emission_matrix[i][j] = float(emission_matrix_count[i][j]) / float(number_of_emission)
                        entire_emission_matrix[i][j] += emission_matrix[i][j] * len(sequence[0])

                matrices.append((transition_matrix, emission_matrix, sequence))
            else:
                print("ERROR: Length of emission sequence is not the same as state sequence!")
                print "len(sequence[0])", len(sequence[0])
                print "len(sequence[1])", len(sequence[1])
                break

        for i in self.transition_matrix:
            for j in self.transition_matrix[i]:
                self.transition_matrix[i][j] = float(self.transition_matrix[i][j] * self.number_of_emissions + entire_transition_matrix[i][j]) / float(self.number_of_emissions + number_of_emissions)

        for i in self.emission_matrix:
            for j in self.emission_matrix[i]:
                self.emission_matrix[i][j] = float(self.emission_matrix[i][j] * self.number_of_emissions + entire_emission_matrix[i][j]) / float(self.number_of_emissions + number_of_emissions)

        self.number_of_emissions += number_of_emissions

        return matrices

    """ TO DO """
    def baum_welch_training(self, sequences):
        """
        Method that trains current hidden Markov model with given sequences emissions using Baum-Welch algorithm

        :param sequences: List of string of emissions.
        """

        transition_matrix = self.transition_matrix
        emission_matrix = self.emission_matrix
        A = self.make_transition_matrix(self.state_alphabet, 0)
        E = self.make_emission_matrix(self.state_alphabet, self.emission_matrix, 0)

        for sequence in sequences:

            forward_matrix, forward_probability = self.forward_probability(emission_matrix, transition_matrix, sequence)
            backward_matrix, backward_probability = self.backward_probability(emission_matrix, transition_matrix, sequence)

            for i in A:
                print i
                for j in A[i]:
                    current_sum = 0
                    for k in xrange(len(sequence) - 2):
                        current_sum += forward_matrix[i[1]][k] * transition_matrix[i][j] * emission_matrix[j][self.get_emission_by_name(sequence[k + 1])] * backward_matrix[j[1]][k + 1]
                    A[i][j] += float(1.0 / forward_probability) * current_sum

            for i in E:
                for j in E[i]:
                    current_sum = 0
                    for k in xrange(len(sequence) - 1):
                        if self.get_emission_by_name(sequence[k]) == j:
                            current_sum += forward_matrix[i[1]][k] * backward_matrix[i[1]][k]
                    E[i][j] += float(1.0 / forward_probability) * current_sum
    """ TO DO"""

    def posterior_decoding(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior decoding algorithm.

        :param emission_sequence: String that represents the given emission sequence.
        :returns:  String that represents the most probable state sequence and probability scores for each state at each emission.
        """

        forward_matrix, forward_probability = self.forward_probability(self.emission_matrix, self.transition_matrix, emission_sequence)
        backward_matrix, backward_probability = self.backward_probability(self.emission_matrix, self.transition_matrix, emission_sequence)
        score_matrix = numpy.zeros(forward_matrix.shape)

        sequence = ""
        for index in xrange(len(emission_sequence) + 1):
            current_max = []
            for i in xrange(len(self.emission_matrix)):
                forward = forward_matrix[i][index] * backward_matrix[i][index]
                current_max.append(forward)
                score_matrix[i][index] = forward
            score_matrix[:, index] = score_matrix[:, index] / numpy.sum(score_matrix[:, index])
            index_of_max = current_max.index(max(current_max))
            sequence += str(self.get_state_by_index(index_of_max)[0].name)

        return numpy.array(list(sequence)), score_matrix

    def posterior_decoding_normalized(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior decoding algorithm that uses normalization.

        :param emission_sequence: String that represents the given emission sequence.
        :returns:  String that represents the most probable state sequence and probability scores for each state at each emission.
        """

        forward_matrix, forward_probability = self.forward_probability_normalized(self.emission_matrix, self.transition_matrix, emission_sequence)
        backward_matrix, backward_probability = self.backward_probability_normalized(self.emission_matrix, self.transition_matrix, emission_sequence)
        score_matrix = numpy.zeros(forward_matrix.shape)

        sequence = ""
        for index in xrange(len(emission_sequence) + 1):
            current_max = []
            for i in xrange(len(self.emission_matrix)):
                forward = forward_matrix[i][index] * backward_matrix[i][index]
                current_max.append(forward)
                score_matrix[i][index] = forward
            score_matrix[:, index] = score_matrix[:, index] / numpy.sum(score_matrix[:, index])
            index_of_max = current_max.index(max(current_max))
            sequence += str(self.get_state_by_index(index_of_max)[0].name)

        return numpy.array(list(sequence)), score_matrix

    def posterior_decoding_constant(self, emission_sequence, constant):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior decoding algorithm that uses constant scaling.

        :param emission_sequence: String that represents the given emission sequence.
        :param constant: Value with which we multiply values at each step to prevent underflow.
        :returns:  String that represents the most probable state sequence and probability scores for each state at each emission.
        """

        forward_matrix, forward_probability = self.forward_probability_constant(self.emission_matrix, self.transition_matrix, emission_sequence, constant)
        backward_matrix, backward_probability = self.backward_probability_constant(self.emission_matrix, self.transition_matrix, emission_sequence, constant)
        score_matrix = numpy.zeros(forward_matrix.shape)

        sequence = ""
        for index in xrange(len(emission_sequence) + 1):
            current_max = []
            for i in xrange(len(self.emission_matrix)):
                forward = forward_matrix[i][index] * backward_matrix[i][index]
                current_max.append(forward)
                score_matrix[i][index] = forward
            score_matrix[:, index] = score_matrix[:, index] / numpy.sum(score_matrix[:, index])
            index_of_max = current_max.index(max(current_max))
            sequence += str(self.get_state_by_index(index_of_max)[0].name)

        return numpy.array(list(sequence)), score_matrix

    def posterior_decoding_scaled(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior decoding algorithm that uses scaling.

        :param emission_sequence: String that represents the given emission sequence.
        :returns: String that represents the most probable state sequence and probability scores for each state at each emission.
        """

        forward_matrix, forward_probability, scale_matrix = self.forward_probability_scaled(self.emission_matrix, self.transition_matrix, emission_sequence)
        backward_matrix, backward_probability = self.backward_probability_scaled(self.emission_matrix, self.transition_matrix, emission_sequence, scale_matrix)
        score_matrix = numpy.zeros(forward_matrix.shape)

        sequence = ""
        for index in xrange(len(emission_sequence) + 1):
            current_max = []
            for i in xrange(len(self.emission_matrix)):
                forward = forward_matrix[i][index] * backward_matrix[i][index]
                current_max.append(forward)
                score_matrix[i][index] = forward
            score_matrix[:, index] = score_matrix[:, index] / numpy.sum(score_matrix[:, index])
            index_of_max = current_max.index(max(current_max))
            sequence += str(self.get_state_by_index(index_of_max)[0].name)

        return numpy.array(list(sequence)), score_matrix

    def posterior_decoding_log(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior decoding algorithm that uses logarithmic values and operation.

        :param emission_sequence: String that represents the given emission sequence.
        :returns: String that represents the most probable state sequence and probability scores for each state at each emission.
        """

        forward_matrix, forward_probability = self.forward_probability_log(self.emission_matrix, self.transition_matrix, emission_sequence)
        backward_matrix, backward_probability = self.backward_probability_log(self.emission_matrix, self.transition_matrix, emission_sequence)
        score_matrix = numpy.zeros(forward_matrix.shape)

        sequence = ""
        for index in xrange(len(emission_sequence) + 1):
            current_max = []
            score_sum = 0
            normalizer = "LOGZERO"
            for i in xrange(len(self.emission_matrix)):
                score_matrix[i][index] = self.elnproduct(forward_matrix[i][index], backward_matrix[i][index])
                normalizer = self.elnsum(normalizer, score_matrix[i][index])
            for i in xrange(len(self.emission_matrix)):
                score_matrix[i][index] = self.elnproduct(score_matrix[i][index], -normalizer)
                current_max.append(score_matrix[i][index])
            index_of_max = current_max.index(max(current_max))
            sequence += str(self.get_state_by_index(index_of_max)[0].name)

        return numpy.array(list(sequence)), score_matrix

    def forward_probability(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the forward matrix and forward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Forward matrix and forward probability.

        1. Initialization:
            Alpha_{t}(i) = Pi_{i} * b_{i}(O_{1}), 1 <= i <= N.
        2. Induction:
            Alpha_{t+1}(j) = [SUM_{i=1}^{N} Alpha_{t}(i) * a_{ij}] * b_{j}(O_{t+1}), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of forward matrix.
        number_of_states = len(self.state_alphabet)
        forward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of first values of forward matrix.
        for i in xrange(len(emission_matrix)):
            forward_matrix[i][0] = (1.0 / float(number_of_states))

        # Calculation of 1 -> T values of forward matrix.
        for index in xrange(1, len(emission_sequence) + 1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += forward_matrix[j][index - 1] * transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)]
                forward_matrix[i][index] = emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index - 1])] * state_sum

        # Calculation of the forward probability.
        forward_probability = 0
        for i in xrange(len(emission_matrix)):
            forward_probability += forward_matrix[i][len(emission_sequence)]

        return forward_matrix, forward_probability

    def forward_probability_normalized(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the forward matrix and forward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Forward matrix and forward probability.

        1. Initialization:
            Alpha_{t}(i) = Pi_{i} * b_{i}(O_{1}), 1 <= i <= N.
        2. Induction:
            Alpha_{t+1}(j) = [SUM_{i=1}^{N} Alpha_{t}(i) * a_{ij}] * b_{j}(O_{t+1}), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of forward matrix.
        number_of_states = len(self.state_alphabet)
        forward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of first values of forward matrix.
        for i in xrange(len(emission_matrix)):
            forward_matrix[i][0] = (1.0 / float(number_of_states))

        # Calculation of 1 -> T values of forward matrix.
        for index in xrange(1, len(emission_sequence) + 1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += forward_matrix[j][index - 1] * transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)]
                forward_matrix[i][index] = emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index - 1])] * state_sum
            forward_matrix[:, index] = forward_matrix[:, index] / numpy.sum(forward_matrix[:, index])

        # Calculation of the forward probability.
        forward_probability = 0
        for i in xrange(len(emission_matrix)):
            forward_probability += forward_matrix[i][len(emission_sequence)]

        return forward_matrix, forward_probability

    def forward_probability_constant(self, emission_matrix, transition_matrix, emission_sequence, constant):
        """
        Method that calculates the forward matrix and forward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :param constant: Value with which we multiply values at each step to prevent underflow.
        :returns:  Forward matrix and forward probability.

        1. Initialization:
            Alpha_{t}(i) = Pi_{i} * b_{i}(O_{1}), 1 <= i <= N.
        2. Induction:
            Alpha_{t+1}(j) = [SUM_{i=1}^{N} Alpha_{t}(i) * a_{ij}] * b_{j}(O_{t+1}), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of forward matrix.
        number_of_states = len(self.state_alphabet)
        forward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of first values of forward matrix.
        for i in xrange(len(emission_matrix)):
            forward_matrix[i][0] = (1.0 / float(number_of_states)) * constant

        # Calculation of 1 -> T values of forward matrix.
        for index in xrange(1, len(emission_sequence) + 1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += forward_matrix[j][index - 1] * transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)]
                forward_matrix[i][index] = emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index - 1])] * state_sum * constant

        # Calculation of the forward probability.
        forward_probability = 0
        for i in xrange(len(emission_matrix)):
            forward_probability += forward_matrix[i][len(emission_sequence)]

        return forward_matrix, forward_probability

    def forward_probability_scaled(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the scaled forward matrix and scaled forward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Forward matrix, scale matrix and forward probability.

        1. Initialization:
            Alpha_{1}(i) = Pi_{i} * b_{i}(O_{1}), 1 <= i <= N.
        2. Induction:
            Alpha_{t+1}(j) = [SUM_{i=1}^{N} Alpha_{t}(i) * a_{ij}] * b_{j}(O_{t+1}), 1 <= t <= T - 1, 1 <= j <= N.

        Scaling:
            1. Initialization:
                Alpha_{1}^{**}(i) = Alpha_{1}(i)
                c_{1} = 1 / [SUM_{i=1}^{N} Alpha_{1}^{**}(i)]
                Alpha_{1}^{^}(i) = c_{1} * Alpha_{1}^{**}(i)
            2. Induction:
                Alpha_{1}^{**}(i) = [SUM_{j=1}^{N} Alpha_{t-1}^{^}(j) * a_{ji}] * b_{i}(O_{t})
                c_{t} = 1 / [SUM_{i=1}^{N} Alpha_{t}^{**}(i)]
                Alpha_{t}^{^}(i) = c_{t} * Alpha_{t}^{**}(i)
        """

        # Initialization of forward and scale matrices.
        number_of_states = len(self.state_alphabet)
        forward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)
        scale_matrix = numpy.zeros(len(emission_sequence) + 1)

        # Calculation of first values of forward matrix.
        scale_sum = 0
        for i in xrange(len(emission_matrix)):
            forward_matrix[i][0] = (1.0 / float(number_of_states))
        scale_matrix[0] = numpy.sum(forward_matrix[:, 0])
        forward_matrix[:, 0] = forward_matrix[:, 0] / numpy.sum(forward_matrix[:, 0])

        # Calculation of 1 -> T values of forward matrix.
        for index in xrange(1, len(emission_sequence) + 1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += forward_matrix[j][index - 1] * transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)]
                forward_matrix[i][index] = emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index - 1])] * state_sum
            scale_matrix[index] = numpy.sum(forward_matrix[:, index])
            forward_matrix[:, index] = forward_matrix[:, index] / numpy.sum(forward_matrix[:, index])

        # Calculation of the forward probability.
        forward_probability = 0
        for i in xrange(len(emission_matrix)):
            forward_probability += forward_matrix[i][len(emission_sequence)]

        return forward_matrix, forward_probability, scale_matrix

    def forward_probability_log(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the forward matrix and forward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Forward matrix and forward probability.

        1. Initialization:
            Alpha_{t}(i) = Pi_{i} * b_{i}(O_{1}), 1 <= i <= N.
        2. Induction:
            Alpha_{t+1}(j) = [SUM_{i=1}^{N} Alpha_{t}(i) * a_{ij}] * b_{j}(O_{t+1}), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of forward matrix.
        number_of_states = len(self.state_alphabet)
        forward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)
        emission_matrix = self.matrix_eln_conversion(emission_matrix)
        transition_matrix = self.matrix_eln_conversion(transition_matrix)

        # Calculation of first values of forward matrix.
        for i in xrange(len(emission_matrix)):
            forward_matrix[i][0] = self.eln(1.0 / float(number_of_states))

        # Calculation of 1 -> T values of forward matrix.
        for index in xrange(1, len(emission_sequence) + 1):
            for i in xrange(len(emission_matrix)):
                state_sum = "LOGZERO"
                for j in xrange(len(emission_matrix)):
                    state_sum = self.elnsum(state_sum, self.elnproduct(forward_matrix[j][index - 1], transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)]))
                forward_matrix[i][index] = self.elnproduct(emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index - 1])], state_sum)

        # Calculation of the forward probability.
        forward_probability = "LOGZERO"
        for i in xrange(len(emission_matrix)):
            forward_probability = self.elnsum(forward_matrix[i][len(emission_sequence)], forward_probability)

        return forward_matrix, forward_probability

    def backward_probability(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the backward matrix and backward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Backward matrix and backward probability.

        1. Initialization:
            Beta_{T}(i) = 1, 1 <= i <= N.
        2. Induction:
            Beta_{t}(i) = SUM_{j=1}^{N} a_{ij} * b_{j}(O_{t+1} * Beta_{t+1}(j)), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of backward matrix.
        number_of_states = len(self.state_alphabet)
        backward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of last values of backward matrix.
        for i in xrange(len(emission_matrix)):
            backward_matrix[i][len(emission_sequence)] = 1.0

        # Calculation of T - 1 -> 0 values of backward matrix.
        for index in xrange(len(emission_sequence) - 1, -1, -1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += backward_matrix[j][index + 1] * transition_matrix[self.get_state_by_index(i)][self.get_state_by_index(j)] * emission_matrix[self.get_state_by_index(j)][self.get_emission_by_name(emission_sequence[index])]
                backward_matrix[i][index] = state_sum

        # Calculation of the backward probability.
        backward_probability = 0
        for i in xrange(len(emission_matrix)):
            backward_probability += (1.0 / float(number_of_states)) * emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])] * backward_matrix[i][0]

        return backward_matrix, backward_probability

    def backward_probability_normalized(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the backward matrix and backward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Backward matrix and backward probability.

        1. Initialization:
            Beta_{T}(i) = 1, 1 <= i <= N.
        2. Induction:
            Beta_{t}(i) = SUM_{j=1}^{N} a_{ij} * b_{j}(O_{t+1} * Beta_{t+1}(j)), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of backward matrix.
        number_of_states = len(self.state_alphabet)
        backward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of last values of backward matrix.
        for i in xrange(len(emission_matrix)):
            backward_matrix[i][len(emission_sequence)] = 1.0

        # Calculation of T - 1 -> 0 values of backward matrix.
        for index in xrange(len(emission_sequence) - 1, -1, -1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += backward_matrix[j][index + 1] * transition_matrix[self.get_state_by_index(i)][self.get_state_by_index(j)] * emission_matrix[self.get_state_by_index(j)][self.get_emission_by_name(emission_sequence[index])]
                backward_matrix[i][index] = state_sum
            backward_matrix[:, index] = backward_matrix[:, index] / numpy.sum(backward_matrix[:, index])

        # Calculation of the backward probability.
        backward_probability = 0
        for i in xrange(len(emission_matrix)):
            backward_probability += (1.0 / float(number_of_states)) * emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])] * backward_matrix[i][0]

        return backward_matrix, backward_probability

    def backward_probability_constant(self, emission_matrix, transition_matrix, emission_sequence, constant):
        """
        Method that calculates the backward matrix and backward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :param constant: Value with which we multiply values at each step to prevent underflow.
        :returns:  Backward matrix and backward probability.

        1. Initialization:
            Beta_{T}(i) = 1, 1 <= i <= N.
        2. Induction:
            Beta_{t}(i) = SUM_{j=1}^{N} a_{ij} * b_{j}(O_{t+1} * Beta_{t+1}(j)), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of backward matrix.
        number_of_states = len(self.state_alphabet)
        backward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of last values of backward matrix.
        for i in xrange(len(emission_matrix)):
            backward_matrix[i][len(emission_sequence)] = 1.0 * constant

        # Calculation of T - 1 -> 0 values of backward matrix.
        for index in xrange(len(emission_sequence) - 1, -1, -1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += backward_matrix[j][index + 1] * transition_matrix[self.get_state_by_index(i)][self.get_state_by_index(j)] * emission_matrix[self.get_state_by_index(j)][self.get_emission_by_name(emission_sequence[index])]
                backward_matrix[i][index] = state_sum * constant

        # Calculation of the backward probability.
        backward_probability = 0
        for i in xrange(len(emission_matrix)):
            backward_probability += (1.0 / float(number_of_states)) * emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])] * backward_matrix[i][0]

        return backward_matrix, backward_probability

    def backward_probability_scaled(self, emission_matrix, transition_matrix, emission_sequence, scale_matrix):
        """
        Method that calculates the backward matrix and backward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :param scale_matrix: Matrix of scales from forward algorithm.
        :returns:  Backward matrix and backward probability.

        1. Initialization:
            Beta_{T}(i) = 1, 1 <= i <= N.
        2. Induction:
            Beta_{t}(i) = SUM_{j=1}^{N} a_{ij} * b_{j}(O_{t+1} * Beta_{t+1}(j)), 1 <= t <= T - 1, 1 <= j <= N.

        Scaling:
            1. Initialization:
                Beta_{T}^{**}(i) = 1
                Beta_{T}^{^}(i) = c_{T} * Beta_{T}^{**}(i)
            2. Induction:
                Beta_{t}^{**}(i) = SUM_{j=1}^{N} a_{ij} * b_{j}(O_{t+1} * Beta_{t+1}^{^}(j))
                Beta_{t}^{^}(i) = c_{t} * Beta_{t}^{**}(i)
        """

        # Initialization of backward matrix.
        number_of_states = len(self.state_alphabet)
        backward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)

        # Calculation of last values of backward matrix.
        for i in xrange(len(emission_matrix)):
            backward_matrix[i][len(emission_sequence)] = 1.0

        # Calculation of T - 1 -> 0 values of backward matrix.
        for index in xrange(len(emission_sequence) - 1, -1, -1):
            for i in xrange(len(emission_matrix)):
                state_sum = 0
                for j in xrange(len(emission_matrix)):
                    state_sum += backward_matrix[j][index + 1] * transition_matrix[self.get_state_by_index(i)][self.get_state_by_index(j)] * emission_matrix[self.get_state_by_index(j)][self.get_emission_by_name(emission_sequence[index])]
                backward_matrix[i][index] = state_sum * float(1.0 / scale_matrix[index])

        # Calculation of the backward probability.
        backward_probability = 0
        for i in xrange(len(emission_matrix)):
            backward_probability += (1.0 / float(number_of_states)) * emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])] * backward_matrix[i][0]

        return backward_matrix, backward_probability

    def backward_probability_log(self, emission_matrix, transition_matrix, emission_sequence):
        """
        Method that calculates the backward matrix and backward probability for give sequence of emissions.

        :param emission_matrix: Represents the current emission matrix of our HMM.
        :param transition_matrix: Represents the current transition matrix of our HMM.
        :param emission_sequence: String that represents the given emission sequence.
        :returns:  Backward matrix and backward probability.

        1. Initialization:
            Beta_{T}(i) = 1, 1 <= i <= N.
        2. Induction:
            Beta_{t}(i) = SUM_{j=1}^{N} a_{ij} * b_{j}(O_{t+1} * Beta_{t+1}(j)), 1 <= t <= T - 1, 1 <= j <= N.
        """

        # Initialization of backward matrix.
        number_of_states = len(self.state_alphabet)
        backward_matrix = numpy.zeros((len(emission_sequence) + 1) * number_of_states).reshape(number_of_states, len(emission_sequence) + 1)
        emission_matrix = self.matrix_eln_conversion(emission_matrix)
        transition_matrix = self.matrix_eln_conversion(transition_matrix)

        # Calculation of last values of backward matrix.
        for i in xrange(len(emission_matrix)):
            backward_matrix[i][len(emission_sequence)] = 0.0

        # Calculation of T - 1 -> 0 values of backward matrix.
        for index in xrange(len(emission_sequence) - 1, -1, -1):
            for i in xrange(len(emission_matrix)):
                state_sum = "LOGZERO"
                for j in xrange(len(emission_matrix)):
                    state_sum = self.elnsum(state_sum, self.elnproduct(transition_matrix[self.get_state_by_index(i)][self.get_state_by_index(j)], self.elnproduct(backward_matrix[j][index + 1], emission_matrix[self.get_state_by_index(j)][self.get_emission_by_name(emission_sequence[index])])))
                backward_matrix[i][index] = state_sum

        # Calculation of the backward probability.
        backward_probability = "LOGZERO"
        for i in xrange(len(emission_matrix)):
            backward_probability = self.elnsum(self.elnproduct(self.eln(1.0 / float(number_of_states)), self.elnproduct(emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])], backward_matrix[i][0])), backward_probability)

        return backward_matrix, backward_probability

    def viterbi_decoding(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with viterbi decoding algorithm.

        :param emission_sequence: String that represents the given emission sequence.
        :returns:  String that represents the most probable state sequence and viterbi probability.
        """

        number_of_states = len(self.state_alphabet)
        viterbi_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))
        pointer_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))

        for i in xrange(len(self.emission_matrix)):
            viterbi_matrix[i][0] = self.emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])] * (1.0 / float(number_of_states))

        for index in xrange(1, len(emission_sequence)):
            for i in xrange(len(self.emission_matrix)):
                current_max = []
                for j in xrange(len(self.emission_matrix)):
                    current_max.append(viterbi_matrix[j][index - 1] * self.transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)])
                viterbi_matrix[i][index] = self.emission_matrix[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index])] * max(current_max)
                pointer_matrix[i][index] = current_max.index(max(current_max))

        current_max = []
        for i in xrange(len(self.emission_matrix)):
            current_max.append(viterbi_matrix[i][len(emission_sequence) - 1])
        viterbi_probability = max(current_max)

        index_of_max = current_max.index(max(current_max))
        sequence = str(self.get_state_by_index(index_of_max)[0].name)
        for index in xrange(len(emission_sequence) - 1, 0, -1):
            sequence += str(self.get_state_by_index(pointer_matrix[index_of_max][index])[0].name)
            index_of_max = pointer_matrix[index_of_max][index]

        return sequence[::-1], viterbi_probability

    def viterbi_decoding_log(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with viterbi decoding algorithm with logarithmic probabilities.

        :param emission_sequence: String that represents the given emission sequence.
        :returns:  String that represents the most probable state sequence and viterbi probability.
        """

        number_of_states = len(self.state_alphabet)
        viterbi_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))
        pointer_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))

        transition_matrix_log = self.matrix_log_conversion(self.transition_matrix)
        emission_matrix_log = self.matrix_log_conversion(self.emission_matrix)

        for i in xrange(len(self.emission_matrix)):
            viterbi_matrix[i][0] = emission_matrix_log[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[0])] + math.log10(1.0) + math.log10(1.0 / float(number_of_states))

        for index in xrange(1, len(emission_sequence)):
            for i in xrange(len(self.emission_matrix)):
                current_max = []
                for j in xrange(len(self.emission_matrix)):
                    current_max.append(viterbi_matrix[j][index - 1] + transition_matrix_log[self.get_state_by_index(j)][self.get_state_by_index(i)])
                viterbi_matrix[i][index] = emission_matrix_log[self.get_state_by_index(i)][self.get_emission_by_name(emission_sequence[index])] + max(current_max)
                pointer_matrix[i][index] = current_max.index(max(current_max))

        current_max = []
        for i in xrange(len(self.emission_matrix)):
            current_max.append(viterbi_matrix[i][len(emission_sequence) - 1])
        viterbi_probability = max(current_max)

        index_of_max = current_max.index(max(current_max))
        sequence = str(self.get_state_by_index(index_of_max)[0].name)
        for index in xrange(len(emission_sequence) - 1, 0, -1):
            sequence += str(self.get_state_by_index(pointer_matrix[index_of_max][index])[0].name)
            index_of_max = pointer_matrix[index_of_max][index]

        return sequence[::-1], viterbi_probability

    def posterior_viterbi(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior viterbi decoding algorithm.

        :param emission_sequence: String that represents the given emission sequence.
        :returns: String that represents the most probable state sequence and posterior viterbi probability.
        """

        forward_matrix, forward_probability = self.forward_probability(self.emission_matrix, self.transition_matrix, emission_sequence)
        backward_matrix, backward_probability = self.backward_probability(self.emission_matrix, self.transition_matrix, emission_sequence)

        number_of_states = len(self.state_alphabet)
        viterbi_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))
        pointer_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))

        for i in xrange(len(self.emission_matrix)):
            viterbi_matrix[i][0] = ((forward_matrix[i][0] * backward_matrix[i][0]) / forward_probability) * (1.0 / float(number_of_states))

        for index in xrange(1, len(emission_sequence)):
            for i in xrange(len(self.emission_matrix)):
                current_max = []
                for j in xrange(len(self.emission_matrix)):
                    if self.transition_matrix[self.get_state_by_index(j)][self.get_state_by_index(i)] > 0.0:
                        current_max.append(viterbi_matrix[j][index - 1])
                viterbi_matrix[i][index] = ((forward_matrix[i][index] * backward_matrix[i][index]) / forward_probability) * max(current_max)
                pointer_matrix[i][index] = current_max.index(max(current_max))

        current_max = []
        for i in xrange(len(self.emission_matrix)):
            current_max.append(viterbi_matrix[i][len(emission_sequence) - 1])
        viterbi_probability = max(current_max)

        index_of_max = current_max.index(max(current_max))
        sequence = str(self.get_state_by_index(index_of_max)[0].name)
        for index in xrange(len(emission_sequence) - 1, 0, -1):
            sequence += str(self.get_state_by_index(pointer_matrix[index_of_max][index])[0].name)
            index_of_max = pointer_matrix[index_of_max][index]

        return sequence[::-1], viterbi_probability

    def posterior_viterbi_log_scaled(self, emission_sequence):
        """
        Method that calculates the most probable path of states for give sequence of emissions with posterior viterbi decoding algorithm with logarithmic and scaled probabilities.

        :param emission_sequence: String that represents the given emission sequence.
        :returns:  String that represents the most probable state sequence and posterior viterbi probability.
        """

        forward_matrix, forward_probability, scale_matrix = self.forward_probability_scaled(self.emission_matrix, self.transition_matrix, emission_sequence)
        backward_matrix, backward_probability = self.backward_probability_scaled(self.emission_matrix, self.transition_matrix, emission_sequence, scale_matrix)

        transition_matrix_log = self.matrix_log_conversion(self.transition_matrix)

        number_of_states = len(self.state_alphabet)
        viterbi_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))
        pointer_matrix = numpy.zeros(len(emission_sequence) * number_of_states).reshape(number_of_states, len(emission_sequence))

        for i in xrange(len(self.emission_matrix)):
            viterbi_matrix[i][0] = math.log10((forward_matrix[i][0] * backward_matrix[i][0]) / forward_probability) + math.log10(1.0) + math.log10(1.0 / float(number_of_states))

        for index in xrange(1, len(emission_sequence)):
            for i in xrange(len(self.emission_matrix)):
                current_max = []
                for j in xrange(len(self.emission_matrix)):
                    if transition_matrix_log[self.get_state_by_index(j)][self.get_state_by_index(i)] > math.log10(0.000000000001):
                        current_max.append(viterbi_matrix[j][index - 1])
                viterbi_matrix[i][index] = math.log10((forward_matrix[i][index] * backward_matrix[i][index]) / forward_probability) + max(current_max)
                pointer_matrix[i][index] = current_max.index(max(current_max))

        current_max = []
        for i in xrange(len(self.emission_matrix)):
            current_max.append(viterbi_matrix[i][len(emission_sequence) - 1])
        viterbi_probability = max(current_max)

        index_of_max = current_max.index(max(current_max))
        sequence = str(self.get_state_by_index(index_of_max)[0].name)
        for index in xrange(len(emission_sequence) - 1, 0, -1):
            sequence += str(self.get_state_by_index(pointer_matrix[index_of_max][index])[0].name)
            index_of_max = pointer_matrix[index_of_max][index]

        return sequence[::-1], viterbi_probability

    def save_hmm_to_pkl(self, file_name):
        """
        Method that saves current HMM to pickle file.

        :param file_name: Given file name to which we save current HMM.
        """

        with open(file_name, 'wb') as f:
            pickle.dump(self, f)