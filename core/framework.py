
import random

from Bio import Alphabet

from Bio.Seq import MutableSeq
from Bio.Seq import Seq

from Bio.HMM import MarkovModel
from Bio.HMM import Trainer
from Bio.HMM import DynamicProgramming


class Framework:
    """
    Framework class for single hidden Markov model evaluation.
    """

    def __init__(self, state_alphabet, emission_alphabet):
        """
        Initialization of Framework and it's hidden Markov model.
        :state_alphabet: Array that represents all possible state symbols.
        :emission_alphabet: Array that represents all possible emission symbols.
        """

        # Initialization of state alphabet.
        self.state_alphabet = Alphabet.Alphabet()
        self.state_alphabet.letters = state_alphabet

        # Initialization of emission alphabet.
        self.emission_alphabet = Alphabet.Alphabet()
        self.emission_alphabet.letters = emission_alphabet

        # Initialization of hidden Markov model parameters.
        hmm_builder = MarkovModel.MarkovModelBuilder(self.state_alphabet, self.emission_alphabet)
        hmm_builder.allow_all_transitions()
        hmm_builder.set_equal_probabilities()

        # Creation of hidden Markov model.
        self.standard_hmm = hmm_builder.get_markov_model()

        # Initialization of hidden Markov model trainer.
        self.trainer = Trainer.KnownStateTrainer(self.standard_hmm)

    def train_hmm(self, emission_sequence, state_sequence):
        """
        This function does something.

        :param name: The name to use.
        :param state: Current state to be in.
        :returns:  int -- the return code.

        """
        if len(emission_sequence) == len(state_sequence):
            # Initialize the training sequence from input data.
            know_training_sequence = Trainer.TrainingSequence(emission_sequence, state_sequence)

            # Train the hidden Markov model.
            self.standard_hmm = self.trainer.train([know_training_sequence])
        else:
            print("ERROR: The length of emission_sequence is not equal to state_sequence!")

    def train_hmm_batch(self, emission_sequences, state_sequences):
        """
        Method that train hidden Markov model on multiple sequences.

        :param emission_sequences: Array of arrays that represent sequence of emission symbols.
        :param state_sequences: Array of arrays that represent sequence of state symbols.
        """

        # Check for equal length of emission_sequences and state_sequences.
        if len(emission_sequences) == len(state_sequences):

            # Initialize array of training sequences and append training sequences generated from input data.
            know_training_sequences = []
            for i in xrange(len(emission_sequences)):
                if len(emission_sequences[i]) == len(state_sequences[i]):
                    know_training_sequences.append(Trainer.TrainingSequence(emission_sequences[i], state_sequences[i]))
                else:
                    print("ERROR: The length of emission_sequence is not equal to state_sequence!")

            # Train the hidden Markov model.
            self.standard_hmm = self.trainer.train(know_training_sequences)
        else:
            print("ERROR: The length of emission_sequences is not equal to state_sequences!")

    def viterbi_prediction(self, emission_sequence):
        """
        Method that predicts the most probable state sequence for give emission sequence.
        @emission_sequence - array that represents sequence of emission symbols.
        """

        # Predict the most probable state sequence for input data.
        predicted_states, sequence_probability = self.standard_hmm.viterbi(emission_sequence, self.state_alphabet)
        return predicted_states

    def posterior_prediction(self, emission_sequence, state_sequence):
        """
        Method that predicts the most probable state sequence for give emission sequence.
        @emission_sequence - array that represents sequence of emission symbols.
        """

        sequence = Trainer.TrainingSequence(Seq(''.join(emission_sequence), self.emission_alphabet), Seq(''.join(state_sequence), self.state_alphabet))

        #print(self.get_transition_probabilities())
        posterior = DynamicProgramming.ScaledDPAlgorithms(self.standard_hmm, sequence)
        #forward_varables, forward_probability = posterior.forward_algorithm()
        #print(self.get_transition_probabilities())
        #posterior = DynamicProgramming.ScaledDPAlgorithms(self.standard_hmm, sequence)
        backward_varables, backward_probability = posterior.backward_algorithm()

        #print(forward_varables, forward_probability)
        print(backward_varables, backward_probability)


        return ""

    def get_transition_probabilities(self):
        """
        Method that returns dictionary of transition probabilities between the states of hidden Markov model.
        """

        return self.standard_hmm.transition_prob

    def get_emission_probabilities(self):
        """
        Method that returns dictionary of emission probabilities for each states of hidden Markov model.
        """

        return self.standard_hmm.emission_prob

    def loaded_dice_roll(self, chance_num, cur_state):
        """
        Generate a loaded dice roll based on the state and a random number
        """

        if cur_state == 'F':
            if chance_num <= (float(1) / float(6)):
                return '1'
            elif chance_num <= (float(2) / float(6)):
                return '2'
            elif chance_num <= (float(3) / float(6)):
                return '3'
            elif chance_num <= (float(4) / float(6)):
                return '4'
            elif chance_num <= (float(5) / float(6)):
                return '5'
            else:
                return '6'
        elif cur_state == 'L':
            if chance_num <= (float(1) / float(10)):
                return '1'
            elif chance_num <= (float(2) / float(10)):
                return '2'
            elif chance_num <= (float(3) / float(10)):
                return '3'
            elif chance_num <= (float(4) / float(10)):
                return '4'
            elif chance_num <= (float(5) / float(10)):
                return '5'
            else:
                return '6'
        else:
            raise ValueError("Unexpected cur_state %s" % cur_state)

    def generate_rolls(self, num_rolls):
        """
        Generate a bunch of rolls corresponding to the casino probabilities.
        Returns:
        o The generate roll sequence
        o The state sequence that generated the roll.
        """

        # Start off in the fair state.
        cur_state = 'F'
        roll_seq = []
        state_seq = []

        # Generate the sequence.
        for roll in range(num_rolls):

            state_seq.append(cur_state)

            # Generate a random number.
            chance_num = random.random()

            # Add on a new roll to the sequence.
            new_roll = self.loaded_dice_roll(chance_num, cur_state)
            roll_seq.append(new_roll)

            # Now give us a chance to switch to a new state.
            chance_num = random.random()
            if cur_state == 'F':
                if chance_num <= .05:
                    cur_state = 'L'
            elif cur_state == 'L':
                if chance_num <= .1:
                    cur_state = 'F'

        return roll_seq, state_seq

    def cross_validate(self, length_of_sequences, number_of_sequences, number_of_train_sequences):
        """
        Method that performs cross validation for random generated sequences.
        @length_of_sequences - value that represents the length of each generated sequence.
        @number_of_sequences - value that represents the number of generated sequences.
        @number_of_train_sequences - value that represents the number of generated sequences that we take for training.
        """

        # Generate random sequences.
        sequences = []
        for i in xrange(number_of_sequences):
            rolls, states = self.generate_rolls(length_of_sequences)
            sequences.append([rolls, states])

        # Generate train and test sequences.
        train_sequences = random.sample(sequences, number_of_train_sequences)
        test_sequences = [p for p in sequences if p not in train_sequences]

        # Train hidden Markov model on train sequences.
        self.train_hmm_batch([p[0] for p in train_sequences], [p[1] for p in train_sequences])

        # Predict state sequences for test sequences and calculate difference to ground truth to calculate accuracy.
        accuracies = []
        for test_sequence in test_sequences:
            predicted_states = list(self.viterbi_prediction(test_sequence[0]))

            different = 0
            for i in xrange(len(predicted_states)):
                if predicted_states[i] != test_sequence[1][i]:
                    different += 1

            accuracy = (1 - float(different) / len(test_sequence[0])) * 100
            accuracies.append(accuracy)

        average_accuracy = sum(accuracies)/len(accuracies)
        #print("Average accuracy of give evaluation is: %.2f%%" % average_accuracy)
        return average_accuracy

    def cross_validate_multiple(self, length_of_sequences, number_of_sequences, number_of_train_sequences, number_of_repetitions):
        """
        Method that performs multiple cross validation for random generated sequences.
        @length_of_sequences - value that represents the length of each generated sequence.
        @number_of_sequences - value that represents the number of generated sequences.
        @number_of_train_sequences - value that represents the number of generated sequences that we take for training.
        @number_of_repetitions -  value that represents the repetition of single cross validation.
        """

        accuracies = []
        for i in xrange(number_of_repetitions):
            current_accuracy = self.cross_validate(length_of_sequences, number_of_sequences, number_of_train_sequences)
            accuracies.append(current_accuracy)

        average_accuracy = sum(accuracies)/len(accuracies)
        print("Average accuracy of give evaluation is: %.2f%%" % average_accuracy)
        return average_accuracy