# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
import scipy.stats as sps
import sklearn.metrics as mt
import numpy as np
import collections as cs
import random as rd
import networkx as nx
import pickle
import gzip
import os
import math
import re
import glob
import TableFactory

from matplotlib.colors import hsv_to_rgb

from iCLIPHMM.core import HiddenMarkovModel
from iCLIPHMM.core import FileManager


class AnalyticsManager:
    """
    Class that represent analytics manager.
    """

    def __init__(self):
        self.id = 0

    @staticmethod
    def create_vector(sequence):
        """
        Method that creates single vector that represents the given sequence from it's transition and emission matrix.

        :param sequence: Given transition and emission matrix.
        :returns: Vector with probabilities.
        """

        vector = []

        transition_matrix = sequence[0]
        for i in transition_matrix:
            for j in transition_matrix[i]:
                vector.append(transition_matrix[i][j])

        emission_matrix = sequence[1]
        for i in emission_matrix:
            for j in emission_matrix[i]:
                vector.append(emission_matrix[i][j])

        return np.array(vector)

    def cross_validation_of_second_stage_hmm2(self, train_sequences, test_sequences, state_neighbourhoods, best_emissions, emissions_probability_ratio, min_in_state_ratio, file_name, hmm_type, dump, evaluation, selected_experiments, iclip_info_file, experiment_id):
        """
        Method that builds second stage HMMs for given parameters and returns the best one.

        :param train_sequences: Given train sequences.
        :param test_sequences: Given test sequences.
        :param state_neighbourhoods: Dictionary which represents the size of neighbourhood for each state.
        :param best_emissions: List of numbers, which represent the amount of the best emissions chosen for hmm building.
        :param emissions_probability_ratio: Minimum appearance probability for chosen emission.
        :param min_in_state_ratio: Minimum in state appearance probability for chosen emission.
        :param file_name: Name of the output files.
        :param hmm_type: String that determines the type of HMM.
        :param dump: Logical switch for writing results to file.
        :param evaluation: String that represents the evaluation method.
        :param selected_experiments: List of selected experiments used.
        :param iclip_info_file: String that represents the name of file, which represents all iCLIP experiment data.
        :param experiment_id: Id of current experiment.
        """

        fm = FileManager()

        iclip_labels = pickle.load(open(iclip_info_file))

        print "Cross validation of second HMM"

        print "|--Counting emissions in inner train data and calculating probabilities:"
        emissons_probabilities = self.second_stage_counts_probabilities(train_sequences, "second_stage_counts_probabilities_" + str(), False)

        print "|--Counting emissions in inner train data for each state and its neighbourhood and calculating probabilities:"
        statistics = self.second_stage_sequence_statistics(train_sequences, state_neighbourhoods, "second_stage_sequence_statistics_" + str(), False)

        ratio = {}
        for key in state_neighbourhoods.keys():
            ratio[key] = {}

        for key in statistics.keys():
            for key2 in statistics[key].keys():
                if statistics[key][key2] > min_in_state_ratio:
                    ratio[key][key2] = (statistics[key][key2] / emissons_probabilities["probabilities"][key2])

        max_value = 0.5
        final_chosen_emissions = [[],[]]

        print "|--Evaluating new sequences:"
        for key in ratio.keys():

            draw_sorted_list = sorted(ratio[key].items(), key=lambda x: x[1], reverse=True)
            self.draw_bar_chart_with_table(draw_sorted_list, 'Emission ratio real_probability/expected_probability (min_in_state_ratio: ' + str(min_in_state_ratio) + ')', "Emissions", "Ratio", "second_stage_visualize_ratio_" + file_name + "_" + key, selected_experiments, iclip_labels, experiment_id, key, min_in_state_ratio)

            output_file = gzip.GzipFile('data_for_table_' + file_name + "_" + key + '.pkl', 'wb')
            output_file.write(pickle.dumps([draw_sorted_list, 'Emission ratio real_probability/expected_probability (min_in_state_ratio: ' + str(min_in_state_ratio) + ')', "Emissions", "Ratio", "second_stage_visualize_ratio_" + file_name + "_" + key, selected_experiments, iclip_labels, experiment_id, key, min_in_state_ratio], -1))
            output_file.close()

            #print draw_sorted_list
            mex_elements = [x for x in draw_sorted_list if x[1] > emissions_probability_ratio]
            #print mex_elements

            for i in best_emissions:
                if i <= len(mex_elements):
                    chosen_emissions = draw_sorted_list[0:i]

                    print "|--|--Evaluating these emissions: ", chosen_emissions

                    generated_extracted_variables = [[j[0] for j in chosen_emissions], []]
                    generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.second_stage_make_sequences(train_sequences, [j[0] for j in chosen_emissions])
                    temp_l = []
                    for seq in [s[1] for s in generated_train_sequences]:
                        temp_l.extend(seq)

                    generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.second_stage_make_sequences(test_sequences, [j[0] for j in chosen_emissions])
                    temp_l = []
                    for seq in [s[1] for s in generated_test_sequences]:
                        temp_l.extend(seq)

                    generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, generated_extracted_variables, generated_test_sequences)

                    results = self.cross_validation_of_hmm(generated_train_sequences, generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, file_name + "_" + key + "_best_" + str(i), dump, evaluation)

                    print "|--|--|--" + evaluation + " result: ", results[0], results[0][key]

                    if results[0][key] > max_value:
                        max_value = results[0][key]
                        final_chosen_emissions = [chosen_emissions, results[0]]

        print "|--Extracted variables: ", final_chosen_emissions[0]
        print "|--Best " + evaluation + " result:", final_chosen_emissions[1]

        return final_chosen_emissions

    def cross_validation_of_second_stage_hmm(self, train_sequences, test_sequences, state_neighbourhoods, best_emissions, emissions_probability_ratio, min_in_state_ratio, file_name, hmm_type, dump, evaluation, selected_experiments, iclip_info_file, experiment_id):
        """
        Method that builds second stage HMMs for given parameters and returns the best one.

        :param train_sequences: Given train sequences.
        :param test_sequences: Given test sequences.
        :param state_neighbourhoods: Dictionary which represents the size of neighbourhood for each state.
        :param best_emissions: List of numbers, which represent the amount of the best emissions chosen for hmm building.
        :param emissions_probability_ratio: Minimum appearance probability for chosen emission.
        :param min_in_state_ratio: Minimum in state appearance probability for chosen emission.
        :param file_name: Name of the output files.
        :param hmm_type: String that determines the type of HMM.
        :param dump: Logical switch for writing results to file.
        :param evaluation: String that represents the evaluation method.
        :param selected_experiments: List of selected experiments used.
        :param iclip_info_file: String that represents the name of file, which represents all iCLIP experiment data.
        :param experiment_id: Id of current experiment.
        """

        fm = FileManager()

        iclip_labels = pickle.load(open(iclip_info_file))

        print "Cross validation of second HMM"

        print "|--Counting emissions in inner train data and calculating probabilities:"
        emissons_probabilities = self.second_stage_counts_probabilities(train_sequences, "test", False)

        print "|--Counting emissions in inner train data for each state and its neighbourhood and calculating probabilities:"
        statistics = self.second_stage_sequence_statistics(train_sequences, state_neighbourhoods, "test", False)

        ratio = {}
        for key in state_neighbourhoods.keys():
            ratio[key] = {}

        for key in statistics.keys():
            for key2 in statistics[key].keys():
                if statistics[key][key2] > min_in_state_ratio:
                    ratio[key][key2] = (statistics[key][key2] / emissons_probabilities["probabilities"][key2])

        max_value = 0.5
        final_chosen_emissions = []

        print "|--Evaluating new sequences:"
        for key in ratio.keys():

            draw_sorted_list = sorted(ratio[key].items(), key=lambda x: x[1], reverse=True)
            self.draw_bar_chart_with_table(draw_sorted_list, 'Emission ratio real_probability/expected_probability (min_in_state_ratio: ' + str(min_in_state_ratio) + ')', "Emissions", "Ratio", "second_stage_visualize_ratio_" + file_name + "_" + key, selected_experiments, iclip_labels, experiment_id, key, min_in_state_ratio)

            for i in best_emissions:
                chosen_emissions = draw_sorted_list[0:i]

                if chosen_emissions[len(chosen_emissions) - 1][1] > emissions_probability_ratio:

                    print "|--|--Evaluating these emissions: ", chosen_emissions

                    generated_extracted_variables = [[j[0] for j in chosen_emissions], []]
                    generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.second_stage_make_sequences(train_sequences, [j[0] for j in chosen_emissions])
                    temp_l = []
                    for seq in [s[1] for s in generated_train_sequences]:
                        temp_l.extend(seq)

                    generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.second_stage_make_sequences(test_sequences, [j[0] for j in chosen_emissions])
                    temp_l = []
                    for seq in [s[1] for s in generated_test_sequences]:
                        temp_l.extend(seq)

                    generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, generated_extracted_variables, generated_test_sequences)

                    results = self.cross_validation_of_hmm(generated_train_sequences, generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, file_name + "_" + key + "_kmers_" + str(i), dump, evaluation)

                    print "|--|--|--" + evaluation + " result: ", results[0], results[0][key]

                    if results[0][key] > max_value:
                        max_value = results[0][key]
                        final_chosen_emissions = [chosen_emissions, results[0]]

        print "|--Extracted variables: ", final_chosen_emissions[0]
        print "|--Best " + evaluation + " result:", final_chosen_emissions[1]

    def cross_validation_of_hmm_with_extraction(self, sequences, percent_of_train_sequences, percent_of_inner_train_sequences, number_of_repetitions, number_of_inner_repetitions, hmm_type, kmer_size, best_kmers, state_neighbourhoods, annotations_size, roc_ratio, kmer_probability_ratio, annotation_probability_ratio, minimal_distance_between_kmer_ratio, minimal_distance_between_annotation_ratio, dump, file_name, evaluation):
        """
        Method that finds the most expressive variables from sequences and build the final HMM.

        :param sequences: List of all sequences.
        :param percent_of_train_sequences: Percent of train data at first division of sequences.
        :param percent_of_inner_train_sequences: Percent of train data at second division of sequences.
        :param number_of_repetitions: Number of repetitions of the whole process.
        :param number_of_inner_repetitions: Number of repetitions of findind the best variables.
        :param hmm_type: String that determines the type of HMM.
        :param kmer_size: Number that denotes the size of kmers.
        :param best_kmers: List of numbers which denotes how many best kmers are used in building HMM.
        :param state_neighbourhoods: Dictionary which represents the size of neighbourhood for each state.
        :param annotations_size: Number that denotes the size of annotations.
        :param roc_ratio: Number that denotes minimal AUC of ROC for variable for use in final HMM building.
        :param kmer_probability_ratio: Number that denotes minimal ratio of kmer for use in HMM building.
        :param annotation_probability_ratio: Number that denotes minimal ratio of annotation for use in HMM building.
        :param minimal_distance_between_kmer_ratio: Number that denotes minimal ratio difference for the same kmer between different states for use in HMM building.
        :param minimal_distance_between_annotation_ratio: Number that denotes minimal ratio difference for the same annotation between different states for use in HMM building.
        :param file_name: Name of the output files.
        :param dump: Logical switch for writing results to file.
        :param evaluation: String that represents the evaluation method.
        """

        number_of_train_sequences = int(len(sequences) * percent_of_train_sequences / 100)

        fm = FileManager()

        print "Cross validation of HMM with extraction: ", file_name

        for iteration in xrange(number_of_repetitions):

            print "Iteration: ", iteration

            # Split sequences between test and train sequences.
            train_ids = rd.sample(range(len(sequences)), number_of_train_sequences)
            test_ids = [p for p in range(len(sequences)) if p not in train_ids]
            train_sequences = [sequences[p] for p in train_ids]
            test_sequences = [sequences[p] for p in test_ids]
            number_of_inner_train_sequences = int(len(train_sequences) * percent_of_inner_train_sequences / 100)

            extracted_variables = []
            for inner_iteration in xrange(number_of_inner_repetitions):

                print "Inner iteration: ", iteration

                # Split train sequences between inner test and inner train sequences.
                inner_train_ids = rd.sample(range(len(train_sequences)), number_of_inner_train_sequences)
                inner_test_ids = [p for p in range(len(train_sequences)) if p not in inner_train_ids]
                inner_train_sequences = [train_sequences[p] for p in inner_train_ids]
                inner_test_sequences = [train_sequences[p] for p in inner_test_ids]

                print "|--Counting nucleotides, kmers and annotations in inner train data and calculating probabilities:"
                n_and_k_probabilities = self.nucleotides_kmers_annotations_counts_probabilities(inner_train_sequences, kmer_size, annotations_size, "inner_iteration_" + file_name, False)

                print "|--Counting nucleotides, kmers and annotations in inner train data for each state and its neighbourhood and calculating probabilities:"
                statistics = self.sequences_statistics(inner_train_sequences, kmer_size, state_neighbourhoods, annotations_size, "inner_iteration_" + file_name, False)

                ratio = {}
                for key in state_neighbourhoods.keys():
                    ratio[key] = {"nucleotides": {}, "kmers": {}, "annotations": {}}

                for key in statistics.keys():
                    for key2 in statistics[key].keys():
                        for key3 in statistics[key][key2].keys():
                            if key2 == "annotations":
                                ratio[key][key2][key3] = (statistics[key][key2][key3] / n_and_k_probabilities["probabilities"][key2][key3]["1"])
                            else:
                                ratio[key][key2][key3] = (statistics[key][key2][key3] / n_and_k_probabilities["probabilities"][key2][key3])

                print "|--Evaluating new sequences:"

                the_chosen_kmers = []
                the_chosen_annotations = []
                for key in ratio.keys():
                    for key2 in ratio[key].keys():

                        if key2 == "kmers":

                            print "|--|--Evaluating k-mers for state " + key + " : "

                            max_value = 0.5
                            max_index = 0
                            sorted_list = sorted(ratio[key][key2].items(), key=lambda x: x[1], reverse=True)

                            if dump:
                                n_kmers = 30
                                draw_sorted_list = sorted_list[0:n_kmers]

                                self.draw_bar_chart(draw_sorted_list, 'K-mer ratio (real probability) / (expected probability)', "K-mers", "Ratio", "visualize_ratio_" + file_name + "_" + key + "_" + str(n_kmers))

                            for i in best_kmers:
                                chosen_kmers = sorted_list[0:i]

                                if chosen_kmers[len(chosen_kmers) - 1][1] > kmer_probability_ratio:

                                    print "|--|--|--Evaluating these k-mers: ", chosen_kmers

                                    generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.make_sequences(inner_train_sequences, kmer_size, [j[0] for j in chosen_kmers], [])
                                    generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.make_sequences(inner_test_sequences, kmer_size, [j[0] for j in chosen_kmers], [])

                                    generated_extracted_variables = [[j[0] for j in chosen_kmers], []]
                                    generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, generated_extracted_variables, generated_test_sequences)

                                    results = self.cross_validation_of_hmm(generated_train_sequences, generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, file_name + "_" + key + "_kmers_" + str(i), dump, evaluation)

                                    print "|--|--|--|--" + evaluation + " result: ", results[0], results[0][key]

                                    if results[0][key] > max_value:
                                        max_value = results[0][key]
                                        max_index = i

                            if max_value > roc_ratio:
                                the_chosen_kmers.extend(sorted_list[0:max_index])

                        elif key2 == "annotations":

                            print "|--|--Evaluating annotations for state " + key + " : "

                            if dump:
                                width = 0.75
                                font_size = 8

                                draw_sorted_list = []
                                for key3 in ratio[key][key2]:
                                    draw_sorted_list.append((int(key3), ratio[key][key2][key3]))

                                draw_sorted_list.sort(key=lambda tup: tup[1], reverse=True)
                                self.draw_bar_chart(draw_sorted_list, 'Annotation ratio (real probability) / (expected probability)', "Annotations", "Ratio", "visualize_ratio_" + file_name + "_" + key + '_annotations')

                            for key3 in ratio[key][key2]:
                                if ratio[key][key2][key3] > annotation_probability_ratio:

                                    print "|--|--|--Evaluating annotation: ", key3

                                    kmers = n_and_k_probabilities["probabilities"]["nucleotides"].keys()
                                    generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.make_sequences(inner_train_sequences, 1, kmers, [int(key3)])
                                    generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.make_sequences(inner_test_sequences, 1, kmers, [int(key3)])

                                    generated_extracted_variables = [kmers, [int(key3)]]
                                    generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, generated_extracted_variables, generated_test_sequences)

                                    results = self.cross_validation_of_hmm(generated_train_sequences, generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, file_name + "_" + key + "_annotations_" + key3, dump, evaluation)

                                    print "|--|--|--|--" + evaluation + " result: ", results[0], results[0][key]

                                    if results[0][key] > roc_ratio:
                                        the_chosen_annotations.append((key3, ratio[key][key2][key3]))

                corrected_kmers = []
                if len(the_chosen_kmers) > 0:

                    kmers = [x[0] for x in the_chosen_kmers]
                    multiple_kmers = set([x for x in kmers if kmers.count(x) > 1])
                    singleton_kmers = set(kmers) - multiple_kmers
                    corrected_kmers.extend(list(singleton_kmers))

                    for kmer in multiple_kmers:

                        ratios = [x[1] for x in the_chosen_kmers if x[0] == kmer]

                        distances = []
                        for i in xrange(len(ratios) - 1):
                            for j in xrange(i + 1, len(ratios)):
                                distances.append(abs(ratios[i] - ratios[j]))

                        if max(distances) > minimal_distance_between_kmer_ratio:
                            corrected_kmers.append(kmer)
                else:
                    corrected_kmers.extend(n_and_k_probabilities["probabilities"]["nucleotides"].keys())

                corrected_annotations = []
                if len(the_chosen_annotations) > 0:

                    annotations = [x[0] for x in the_chosen_annotations]
                    multiple_annotations = set([x for x in annotations if annotations.count(x) > 1])
                    singleton_annotations = set(annotations) - multiple_annotations
                    corrected_annotations.extend(list(singleton_annotations))

                    for annotation in multiple_annotations:

                        ratios = [x[1] for x in the_chosen_annotations if x[0] == annotation]

                        distances = []
                        for i in xrange(len(ratios) - 1):
                            for j in xrange(i + 1, len(ratios)):
                                distances.append(abs(ratios[i] - ratios[j]))

                        if max(distances) > minimal_distance_between_annotation_ratio:
                            corrected_annotations.append(annotation)

                corrected_annotations = [int(x) for x in corrected_annotations]

                extracted_variables = [corrected_kmers, corrected_annotations]

            print "|--|--Extracted variables: " + str(extracted_variables)
            print "|--Evaluating final test sequences:"

            final_generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.make_sequences(train_sequences, len(extracted_variables[0][0]), extracted_variables[0], extracted_variables[1])
            final_generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.make_sequences(test_sequences, len(extracted_variables[0][0]), extracted_variables[0], extracted_variables[1])

            final_generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, extracted_variables, final_generated_test_sequences)

            if dump:
                output_file = gzip.GzipFile('final_generated_sequences_' + file_name + '.pkl', 'wb')
                output_file.write(pickle.dumps([final_generated_train_sequences, final_generated_test_sequences, states_alphabet_train, emissions_alphabet_train, extracted_variables], -1))
                output_file.close()

            results = self.cross_validation_of_hmm(final_generated_train_sequences, final_generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, file_name + "_final", dump, evaluation)
            print "|--|--" + evaluation + " result: ", results[0]

    def cross_validation_of_hmm_with_extraction2(self, sequences, percent_of_train_sequences, percent_of_inner_train_sequences, number_of_repetitions, number_of_inner_repetitions, hmm_type, kmer_size, best_kmers, state_neighbourhoods, annotations_size, roc_ratio, min_in_state_ratio, kmer_probability_ratio, annotation_probability_ratio, minimal_distance_between_kmer_ratio, minimal_distance_between_annotation_ratio, dump, file_name, evaluation):
        """
        Method that finds the most expressive variables from sequences and build the final HMM.

        :param sequences: List of all sequences.
        :param percent_of_train_sequences: Percent of train data at first division of sequences.
        :param percent_of_inner_train_sequences: Percent of train data at second division of sequences.
        :param number_of_repetitions: Number of repetitions of the whole process.
        :param number_of_inner_repetitions: Number of repetitions of findind the best variables.
        :param hmm_type: String that determines the type of HMM.
        :param kmer_size: Number that denotes the size of kmers.
        :param best_kmers: List of numbers which denotes how many best kmers are used in building HMM.
        :param state_neighbourhoods: Dictionary which represents the size of neighbourhood for each state.
        :param annotations_size: Number that denotes the size of annotations.
        :param roc_ratio: Number that denotes minimal AUC of ROC for variable for use in final HMM building.
        :param min_in_state_ratio: Minimum in state appearance probability for chosen emission.
        :param kmer_probability_ratio: Number that denotes minimal ratio of kmer for use in HMM building.
        :param annotation_probability_ratio: Number that denotes minimal ratio of annotation for use in HMM building.
        :param minimal_distance_between_kmer_ratio: Number that denotes minimal ratio difference for the same kmer between different states for use in HMM building.
        :param minimal_distance_between_annotation_ratio: Number that denotes minimal ratio difference for the same annotation between different states for use in HMM building.
        :param file_name: Name of the output files.
        :param dump: Logical switch for writing results to file.
        :param evaluation: String that represents the evaluation method.
        """

        number_of_train_sequences = int(len(sequences) * percent_of_train_sequences / 100)
        the_nucleotides = set()

        fm = FileManager()

        print "Cross validation of HMM with extraction: ", file_name

        for iteration in xrange(number_of_repetitions):

            print "Iteration: ", iteration

            # Split sequences between test and train sequences.
            train_ids = rd.sample(range(len(sequences)), number_of_train_sequences)
            test_ids = [p for p in range(len(sequences)) if p not in train_ids]
            train_sequences = [sequences[p] for p in train_ids]
            test_sequences = [sequences[p] for p in test_ids]
            number_of_inner_train_sequences = int(len(train_sequences) * percent_of_inner_train_sequences / 100)

            extracted_variables = []
            for inner_iteration in xrange(number_of_inner_repetitions):

                print "Inner iteration: ", inner_iteration
                inner_file_name = file_name + "_iter-" + str(inner_iteration)

                # Split train sequences between inner test and inner train sequences.
                inner_train_ids = rd.sample(range(len(train_sequences)), number_of_inner_train_sequences)
                inner_test_ids = [p for p in range(len(train_sequences)) if p not in inner_train_ids]
                inner_train_sequences = [train_sequences[p] for p in inner_train_ids]
                inner_test_sequences = [train_sequences[p] for p in inner_test_ids]

                print "|--Counting nucleotides, kmers and annotations in inner train data and calculating probabilities:"
                n_and_k_probabilities = self.nucleotides_kmers_annotations_counts_probabilities(inner_train_sequences, kmer_size, annotations_size, inner_file_name, False)

                print "|--Counting nucleotides, kmers and annotations in inner train data for each state and its neighbourhood and calculating probabilities:"
                statistics = self.sequences_statistics(inner_train_sequences, kmer_size, state_neighbourhoods, annotations_size, inner_file_name, False)

                print n_and_k_probabilities["probabilities"]["nucleotides"].keys()
                for nuc in n_and_k_probabilities["probabilities"]["nucleotides"].keys():
                    the_nucleotides.add(nuc)

                ratio = {}
                for key in state_neighbourhoods.keys():
                    ratio[key] = {"nucleotides": {}, "kmers": {}, "annotations": {}}

                for key in statistics.keys():
                    for key2 in statistics[key].keys():
                        for key3 in statistics[key][key2].keys():
                            if statistics[key][key2][key3] > min_in_state_ratio:
                                if key2 == "annotations":
                                    ratio[key][key2][key3] = (statistics[key][key2][key3] / n_and_k_probabilities["probabilities"][key2][key3]["1"])
                                else:
                                    ratio[key][key2][key3] = (statistics[key][key2][key3] / n_and_k_probabilities["probabilities"][key2][key3])

                print "|--Evaluating new sequences:"

                the_chosen_kmers = []
                the_chosen_annotations = []
                for key in ratio.keys():
                    for key2 in ratio[key].keys():

                        if key2 == "kmers":

                            print "|--|--Evaluating k-mers for state " + key + " : "

                            max_value = 0.5
                            max_index = 0
                            sorted_list = sorted(ratio[key][key2].items(), key=lambda x: x[1], reverse=True)

                            if len(sorted_list) > 0:
                                if dump:
                                    n_kmers = 30
                                    draw_sorted_list = sorted_list[0:n_kmers]

                                    self.draw_bar_chart(draw_sorted_list, 'K-mer ratio (real probability) / (expected probability)', "K-mers", "Ratio", "visualize_ratio_" + inner_file_name + "_" + key + "_" + str(n_kmers))

                                for i in best_kmers:
                                    chosen_kmers = sorted_list[0:i]

                                    if chosen_kmers[len(chosen_kmers) - 1][1] > kmer_probability_ratio:

                                        print "|--|--|--Evaluating these k-mers: ", chosen_kmers

                                        generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.make_sequences(inner_train_sequences, kmer_size, [j[0] for j in chosen_kmers], [])
                                        generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.make_sequences(inner_test_sequences, kmer_size, [j[0] for j in chosen_kmers], [])

                                        states_alphabet_train.extend(states_alphabet_test)
                                        states_alphabet_train = list(set(states_alphabet_train))

                                        generated_extracted_variables = [[j[0] for j in chosen_kmers], []]
                                        generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, generated_extracted_variables, generated_test_sequences)

                                        results = self.cross_validation_of_hmm(generated_train_sequences, generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, inner_file_name + "_" + key + "_kmers_" + str(i), dump, evaluation)

                                        print "|--|--|--|--" + evaluation + " result: ", results[0], results[0][key]

                                        if results[0][key] > max_value:
                                            max_value = results[0][key]
                                            max_index = i

                                if max_value > roc_ratio:
                                    the_chosen_kmers.extend(sorted_list[0:max_index])

                        elif key2 == "annotations":

                            print "|--|--Evaluating annotations for state " + key + " : "

                            draw_sorted_list = []
                            for key3 in ratio[key][key2]:
                                draw_sorted_list.append((int(key3), ratio[key][key2][key3]))

                            if len(draw_sorted_list) > 0:
                                if dump:
                                    width = 0.75
                                    font_size = 8

                                    draw_sorted_list.sort(key=lambda tup: tup[1], reverse=True)
                                    self.draw_bar_chart(draw_sorted_list, 'Annotation ratio (real probability) / (expected probability)', "Annotations", "Ratio", "visualize_ratio_" + inner_file_name + "_" + key + '_annotations')

                                for key3 in ratio[key][key2]:
                                    if ratio[key][key2][key3] > annotation_probability_ratio:

                                        print "|--|--|--Evaluating annotation: ", key3

                                        kmers = n_and_k_probabilities["probabilities"]["nucleotides"].keys()
                                        generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.make_sequences(inner_train_sequences, 1, kmers, [int(key3)])
                                        generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.make_sequences(inner_test_sequences, 1, kmers, [int(key3)])

                                        states_alphabet_train.extend(states_alphabet_test)
                                        states_alphabet_train = list(set(states_alphabet_train))

                                        generated_extracted_variables = [kmers, [int(key3)]]
                                        generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, generated_extracted_variables, generated_test_sequences)

                                        results = self.cross_validation_of_hmm(generated_train_sequences, generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, inner_file_name + "_" + key + "_annotations_" + key3, dump, evaluation)

                                        print "|--|--|--|--" + evaluation + " result: ", results[0], results[0][key]

                                        if results[0][key] > roc_ratio:
                                            the_chosen_annotations.append((key3, ratio[key][key2][key3]))

                corrected_kmers = []
                print the_chosen_kmers
                if len(the_chosen_kmers) > 0:

                    kmers = [x[0] for x in the_chosen_kmers]
                    multiple_kmers = set([x for x in kmers if kmers.count(x) > 1])
                    singleton_kmers = set(kmers) - multiple_kmers
                    corrected_kmers.extend(list(singleton_kmers))

                    for kmer in multiple_kmers:

                        ratios = [x[1] for x in the_chosen_kmers if x[0] == kmer]

                        distances = []
                        for i in xrange(len(ratios) - 1):
                            for j in xrange(i + 1, len(ratios)):
                                distances.append(abs(ratios[i] - ratios[j]))

                        if max(distances) > minimal_distance_between_kmer_ratio:
                            corrected_kmers.append(kmer)
                else:
                    corrected_kmers.extend(n_and_k_probabilities["probabilities"]["nucleotides"].keys())

                corrected_annotations = []
                print the_chosen_annotations
                if len(the_chosen_annotations) > 0:

                    annotations = [x[0] for x in the_chosen_annotations]
                    multiple_annotations = set([x for x in annotations if annotations.count(x) > 1])
                    singleton_annotations = set(annotations) - multiple_annotations
                    corrected_annotations.extend(list(singleton_annotations))

                    for annotation in multiple_annotations:

                        ratios = [x[1] for x in the_chosen_annotations if x[0] == annotation]

                        distances = []
                        for i in xrange(len(ratios) - 1):
                            for j in xrange(i + 1, len(ratios)):
                                distances.append(abs(ratios[i] - ratios[j]))

                        if max(distances) > minimal_distance_between_annotation_ratio:
                            corrected_annotations.append(annotation)

                corrected_annotations = [int(x) for x in corrected_annotations]

                print [corrected_kmers, corrected_annotations]
                extracted_variables.append([corrected_kmers, corrected_annotations])

            print extracted_variables
            final_extracted_variables = [set(), set()]
            for e_variables in extracted_variables:
                final_extracted_variables[0].update(e_variables[0])
                final_extracted_variables[1].update(e_variables[1])

            print final_extracted_variables

            print len(list(final_extracted_variables[0]))

            final_extracted_variables_list = [[], []]
            if len(list(final_extracted_variables[0])) > 0:
                final_extracted_variables_list[0] = list(final_extracted_variables[0])
            else:
                final_extracted_variables_list[0] = list(the_nucleotides)
            if len(list(final_extracted_variables[1])) > 0:
                final_extracted_variables_list[1] = list(final_extracted_variables[1])

            print "|--|--Extracted variables: " + str(final_extracted_variables_list)
            print "|--Evaluating final test sequences:"

            final_generated_train_sequences, emissions_alphabet_train, states_alphabet_train = fm.make_sequences(train_sequences, len(final_extracted_variables_list[0][0]), final_extracted_variables_list[0], final_extracted_variables_list[1])
            final_generated_test_sequences, emissions_alphabet_test, states_alphabet_test = fm.make_sequences(test_sequences, len(final_extracted_variables_list[0][0]), final_extracted_variables_list[0], final_extracted_variables_list[1])

            states_alphabet_train.extend(states_alphabet_test)
            states_alphabet_train = list(set(states_alphabet_train))

            final_generated_test_sequences = self.check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, final_extracted_variables_list, final_generated_test_sequences)

            if dump:
                output_file = gzip.GzipFile('final_generated_sequences_' + inner_file_name + '.pkl', 'wb')
                output_file.write(pickle.dumps([final_generated_train_sequences, final_generated_test_sequences, states_alphabet_train, emissions_alphabet_train, final_extracted_variables_list], -1))
                output_file.close()

            results = self.cross_validation_of_hmm(final_generated_train_sequences, final_generated_test_sequences, states_alphabet_train, emissions_alphabet_train, hmm_type, inner_file_name + "_final", dump, evaluation)
            print "|--|--" + evaluation + " result: ", results[0]

    def cross_validation_of_hmm(self, train_sequences, test_sequences, states_alphabet, emissions_alphabet, hmm_type, file_name, dump, evaluation):
        """
        Method that builds HMM from train sequences and cross validates it on test sequences.

        :param train_sequences: List of train sequences.
        :param test_sequences: List of test sequences.
        :param states_alphabet: List of state names for HMM.
        :param emissions_alphabet: List of emissions names for HMM.
        :param hmm_type: String that determines the type of HMM.
        :param file_name: Name of the output files.
        :param dump: Logical switch for writing results to file.
        :param evaluation: String that represents the evaluation method.
        :returns: List of ROC values for visualization and singe AUC value for each state.
        """

        hmm = HiddenMarkovModel(emissions_alphabet, states_alphabet, hmm_type)
        hmm.state_sequence_training(train_sequences)

        scores = np.array([])
        tests = np.array([])

        for sequence in test_sequences:
            if sequence[0].shape[0] > 0:

                predicted_sequence, y_score = hmm.posterior_decoding_scaled(sequence[1])
                y_test = hmm.create_state_matrix(sequence)

                if scores.shape[0] == 0:
                    scores = y_score
                else:
                    scores = np.concatenate((scores, y_score), axis=1)

                if tests.shape[0] == 0:
                    tests = y_test
                else:
                    tests = np.concatenate((tests, y_test), axis=1)

        results = []
        if evaluation == "ROC":
            n_classes = len(states_alphabet)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in xrange(n_classes):
                fpr[hmm.get_state_by_index(i)[0].name], tpr[hmm.get_state_by_index(i)[0].name], _ = mt.roc_curve(tests[i, :], scores[i, :])
                roc_auc[hmm.get_state_by_index(i)[0].name] = mt.auc(fpr[hmm.get_state_by_index(i)[0].name], tpr[hmm.get_state_by_index(i)[0].name])

            if dump:
                output_file = gzip.GzipFile('hmm_' + file_name + '.pkl', 'wb')
                output_file.write(pickle.dumps([hmm, roc_auc, fpr, tpr], -1))
                output_file.close()
                self.draw_roc_curve(fpr, tpr, roc_auc, hmm, 'ROC curve and AUC for given HMM', 'False Positive Rate', 'True Positive Rate', "cross_validation_of_hmm_" + file_name)
            results = [roc_auc, fpr, tpr]

        elif evaluation == "MAE":
            n_classes = len(states_alphabet)
            mae = dict()
            for i in xrange(n_classes):
                mae[hmm.get_state_by_index(i)[0].name] = 1.0 - mt.mean_absolute_error(tests[i, :], scores[i, :])

            if dump:
                output_file = gzip.GzipFile('hmm_' + file_name + '.pkl', 'wb')
                output_file.write(pickle.dumps([hmm, mae], -1))
                output_file.close()
            results = [mae]

        elif evaluation == "PCC":
            n_classes = len(states_alphabet)
            pcc = dict()
            for i in xrange(n_classes):
                m_pred = sum(scores[i, :]) / float(scores.shape[1])
                m_true = sum(tests[i, :]) / float(tests.shape[1])


                upper_sum = 0.0
                x_sqrt = 0.0
                y_sqrt = 0.0
                for j in xrange(tests.shape[1]):
                    upper_sum += (scores[i, j] - m_pred) * (tests[i, j] - m_true)
                    x_sqrt += (scores[i, j] - m_pred) * (scores[i, j] - m_pred)
                    y_sqrt += (tests[i, j] - m_true) * (tests[i, j] - m_true)

                pcc[hmm.get_state_by_index(i)[0].name] = upper_sum / (math.sqrt(x_sqrt) * math.sqrt(y_sqrt))

            if dump:
                output_file = gzip.GzipFile('hmm_' + file_name + '.pkl', 'wb')
                output_file.write(pickle.dumps([hmm, pcc], -1))
                output_file.close()
            results = [pcc]

        return results

    def calculate_the_percentage_of_variance_measure(self, matrices, k_range, output_file):
        """
        Method that calculates percentage of variance measure for different number of cluster for k-means clustering on given data and draws them on a graph.

        :param matrices: Given transition and emission matrices.
        :param k_range: Given range of number of cluster to evaluate.
        :param output_file: Given string for naming the output file.
        """

        data = []
        for matrix in matrices:
            data.append(self.create_vector(matrix))
        data = np.array(data)

        k_means = [vq.kmeans(data, k) for k in k_range]
        centroids = [cent for (cent, var) in k_means]

        variances = [vq.vq(data, cent) for cent in centroids]
        average_variances = [sum(dist)/data.shape[0] for (cIdx, dist) in variances]

        self.draw_plot(k_range, average_variances, 'Elbow for KMeans clustering', 'Number of clusters', 'Average within-cluster sum of squares', "calculate_the_percentage_of_variance_measure_" + output_file)

    def clustering_of_matrices(self, matrices, output_file, number_of_clusters):
        """
        Method that clusters given matrices with k-means algorithm and for given number of clusters and represents the results in heat map.

        :param matrices: Given transition and emission matrices.
        :param output_file: Given string for naming the output file.
        :param number_of_clusters: Given number of cluster.
        """

        data = []
        for matrix in matrices:
            data.append(self.create_vector(matrix))
        data = np.array(data)

        k_means = vq.kmeans(data, number_of_clusters)
        clusters, _ = vq.vq(data, k_means[0])
        counts = cs.Counter(clusters)

        header = []
        transition_matrix = matrices[0][0]
        for i in transition_matrix:
            for j in transition_matrix[i]:
                header.append("S: " + str(i[0].name) + " -> " + str(j[0].name))

        emission_matrix = matrices[0][1]
        for i in emission_matrix:
            for j in emission_matrix[i]:
                header.append("E: " + str(i[0].name) + " -> " + str(j[0].name))

        print_header = "".join(['{:>20}'.format(i + "|") for i in header])
        for i in xrange(number_of_clusters):
            print "Cluster " + str(i) + ": " + str(counts[i])
            print print_header
            print "".join(["-" for _ in xrange(len(print_header))])
            print "".join(['{:>20}'.format(str(i) + "|") for i in k_means[0][i]])

        self.draw_heatmap(k_means, number_of_clusters, counts, data, header, "Heatmap of clustersof sequences", "Transitions and emissions", "Cluster centroid", "clustering_of_matrice_" + output_file)

    def sequences_statistics(self, sequences, kmer_size, state_neighbourhoods, annotations_size, file_name, dump):
        """
        Method that calculates the probabilities of nucleotides, kmers and annotations for each specific state and its neighbourhood.

        :param sequences: Given list of sequences.
        :param kmer_size: Given size of kmers.
        :param state_neighbourhoods: Dictionary that represent the neighbourhood size for each specific state.
        :param annotations_size: Number of annotations in emissions.
        :param file_name: Given file name for saving the statistics to disk.
        :param dump: Logical switch for writing results to file.
        :returns: List of probabilities of nucleotides annotations and kmers for given states and neighbourhoods for given list of sequences.
        """

        counts = {}
        for key in state_neighbourhoods.keys():
            counts[key] = {"nucleotides": {}, "kmers": {}, "annotations": {}}

        lengths = {}
        for key in state_neighbourhoods.keys():
            lengths[key] = {"nucleotides": 0, "kmers": 0, "annotations": 0}

        for sequence in sequences:
            sequence_counts = self.count_occurrences(sequence, kmer_size, state_neighbourhoods, annotations_size)
            for key in sequence_counts.keys():
                for key2 in sequence_counts[key].keys():
                    for key3 in sequence_counts[key][key2].keys():
                        if key3 in counts[key][key2].keys():
                            counts[key][key2][key3] += sequence_counts[key][key2][key3]
                        else:
                            counts[key][key2][key3] = sequence_counts[key][key2][key3]
            for key in state_neighbourhoods.keys():
                for key2 in lengths[key]:
                    if key2 == "annotations":
                        lengths[key][key2] += sum(sequence_counts[key]["nucleotides"].values())
                    else:
                        lengths[key][key2] += sum(sequence_counts[key][key2].values())

        for key in counts.keys():
                for key2 in counts[key].keys():
                    for key3 in counts[key][key2].keys():
                        counts[key][key2][key3] /= float(lengths[key][key2])

        if dump:
            with open("sequence_statistics_" + file_name + "_" + str(kmer_size) + '.pkl', 'wb') as f:
                pickle.dump(counts, f)

        return counts

    def second_stage_sequence_statistics(self, sequences, state_neighbourhoods, file_name, dump):
        """
        Method that calculates the probabilities of emissions for each specific state and its neighbourhood.

        :param sequences: Given list of sequences.
        :param state_neighbourhoods: Dictionary that represent the neighbourhood size for each specific state.
        :param file_name: Given file name for saving the statistics to disk.
        :param dump: Logical switch for writing results to file.
        :returns: List of probabilities of emissions for given states and neighbourhoods for given list of sequences.
        """

        counts = {}
        for key in state_neighbourhoods.keys():
            counts[key] = {}

        lengths = {}
        for key in state_neighbourhoods.keys():
            lengths[key] = 0

        for sequence in sequences:
            sequence_counts = self.second_stage_count_occurrences(sequence, state_neighbourhoods)
            for key in sequence_counts.keys():
                for key2 in sequence_counts[key].keys():
                    if key2 in counts[key].keys():
                        counts[key][key2] += sequence_counts[key][key2]
                    else:
                        counts[key][key2] = sequence_counts[key][key2]
            for key in state_neighbourhoods.keys():
                lengths[key] += sum(sequence_counts[key].values())

        for key in counts.keys():
            for key2 in counts[key].keys():
                counts[key][key2] /= float(lengths[key])

        if dump:
            with open("second_stage_sequence_statistics" + file_name + '.pkl', 'wb') as f:
                pickle.dump(counts, f)

        return counts

    def distribution_of_iclip_intensities(self, file_manager, list_of_experiments):
        """
        Method that draws graph of iCLIP intensities for given experiment.

        :param file_manager: Given File Manager object.
        :param list_of_experiments: Given list of experiments.
        :returns: Dictionary of intensities and their counts.
        """

        distribution = dict()

        for experiment in list_of_experiments:

            print "Experiment: " + str(experiment) + ": "

            distribution[experiment] = dict()

            for fi, f in enumerate(glob.glob(file_manager.iclip_files_path + file_manager.iclip_file_name)):

                (iclip_a, _), iclip_labels = pickle.load(gzip.open(f))
                iclip = iclip_a[:, experiment].toarray()
                counts_iclip = cs.Counter(iclip.flatten().tolist())

                for key in counts_iclip.keys():
                    if key in distribution[experiment].keys():
                        distribution[experiment][key] += counts_iclip[key]
                    else:
                        distribution[experiment][key] = counts_iclip[key]

            draw_sorted_list = distribution[experiment].items()
            draw_sorted_list.sort(key=lambda tup: tup[0], reverse=False)
            self.draw_bar_chart(draw_sorted_list[1::], "Distribution of iCLIP intensities", "Intensity", "Counts", "distribution_of_iclip_intensities_" + str(experiment))
        return distribution

    @staticmethod
    def second_stage_count_occurrences(sequence, state_neighbourhoods):
        """
        Method counts occurrences of emissions in neighbourhood for each state.

        :param sequence: Given sequence of states and emissions.
        :param state_neighbourhoods: Dictionary that represents neighbourhood size for each state.
        :returns: List of counts of emissions for given states and neighbourhoods for given sequence.
        """

        counts = {}
        for key in state_neighbourhoods.keys():
            counts[key] = {}

        blob_start = 0
        blob_end = 0
        blob_value = sequence[0][0]
        for index in xrange(1, len(sequence[0])):
            if sequence[0][index] != blob_value or index == len(sequence[0]) - 1:

                if index == len(sequence[0]) - 1:
                    blob_end = index

                blob_start = blob_start - state_neighbourhoods[blob_value]
                if blob_start < 0:
                    blob_start = 0

                blob_end = blob_end + state_neighbourhoods[blob_value]
                if blob_end > (len(sequence[1]) - 1):
                    blob_end = len(sequence[1]) - 1

                neighbourhood = sequence[1][blob_start: blob_end + 1]
                emissions = cs.Counter(neighbourhood)
                for key in emissions.keys():
                    if key in counts[blob_value].keys():
                        counts[blob_value][key] += emissions[key]
                    else:
                        counts[blob_value][key] = emissions[key]

                blob_value = sequence[0][index]
                blob_start = index
                blob_end = index
            else:
                blob_end = index

        return counts

    @staticmethod
    def count_occurrences(sequence, kmer_size, state_neighbourhoods, annotations_size):
        """
        Method counts occurrences of nucleotides, kmers and annotations in neighbourhood for each state.

        :param sequence: Given sequence of states and emissions.
        :param kmer_size: Size of kmers.
        :param state_neighbourhoods: Dictionary that represents neighbourhood size for each state.
        :param annotations_size: Number of annotations in emissions.
        :returns: List of counts of nucleotides annotations and kmers for given states and neighbourhoods for given sequence.
        """

        counts = {}
        for key in state_neighbourhoods.keys():
            counts[key] = {"nucleotides": {}, "kmers": {}, "annotations": {}}

        blob_start = 0
        blob_end = 0
        blob_value = sequence[0][0]
        for index in xrange(1, len(sequence[0])):
            if sequence[0][index] != blob_value or index == len(sequence[0]) - 1:

                if index == len(sequence[0]) - 1:
                    blob_end = index

                blob_start = blob_start - state_neighbourhoods[blob_value]
                if blob_start < 0:
                    blob_start = 0

                blob_end = blob_end + state_neighbourhoods[blob_value]
                if blob_end > (len(sequence[1]) - 1):
                    blob_end = len(sequence[1]) - 1

                neighbourhood = sequence[1][blob_start: blob_end + 1]
                nucleotides = cs.Counter([j[0] for j in neighbourhood])
                for key in nucleotides.keys():
                    if key in counts[blob_value]["nucleotides"].keys():
                        counts[blob_value]["nucleotides"][key] += nucleotides[key]
                    else:
                        counts[blob_value]["nucleotides"][key] = nucleotides[key]

                for i in xrange(annotations_size):
                    annotations = cs.Counter([j[i + 1] for j in neighbourhood])
                    if str(i) in counts[blob_value]["annotations"].keys():
                        counts[blob_value]["annotations"][str(i)] += annotations["1"]
                    else:
                        counts[blob_value]["annotations"][str(i)] = annotations["1"]

                kmers = cs.Counter([''.join([j[0] for j in neighbourhood[i: i + kmer_size]]) for i in range(len(neighbourhood) - kmer_size + 1)])
                for key in kmers.keys():
                    if key in counts[blob_value]["kmers"].keys():
                        counts[blob_value]["kmers"][key] += kmers[key]
                    else:
                        counts[blob_value]["kmers"][key] = kmers[key]

                blob_value = sequence[0][index]
                blob_start = index
                blob_end = index
            else:
                blob_end = index

        return counts

    @staticmethod
    def second_stage_counts_probabilities(sequences, file_name, dump):
        """
        Method that counts the occurrences of emissions and calculates their probabilities.

        :param sequences: List of sequences.
        :param file_name: Name of the output file.
        :param dump: Logical switch for writing results to file.
        :returns: Dictionary of counts and probabilities of emissions.
        """

        statistics = {"counts": {}, "probabilities": {}}

        emissions_length = 0

        for sequence in sequences:

            emissions = cs.Counter(sequence[1])
            emissions_length += len(sequence[1])
            for key in emissions.keys():
                if key in statistics["counts"].keys():
                    statistics["counts"][key] += emissions[key]
                else:
                    statistics["counts"][key] = emissions[key]

        for key in statistics["counts"].keys():
            statistics["probabilities"][key] = float(statistics["counts"][key]) / float(emissions_length)

        if dump:
            with open("second_stage_counts_probabilities_" + file_name + '.pkl', 'wb') as f:
                pickle.dump(statistics, f)

        return statistics

    @staticmethod
    def nucleotides_kmers_annotations_counts_probabilities(sequences, kmer_size, annotations_size, file_name, dump):
        """
        Method that counts the occurrences of nucleotides, kmers and annotations of given size and calculates their probabilities.

        :param sequences: List of sequences.
        :param kmer_size: Size of kmers.
        :param annotations_size: Number of annotations in emissions.
        :param file_name: Name of the output file.
        :param dump: Logical switch for writing results to file.
        :returns: Dictionary of counts and probabilities of nucleotides, kmers and annotations.
        """

        statistics = {"counts": {"nucleotides": {}, "kmers": {}, "annotations": {}}, "probabilities": {"nucleotides": {}, "kmers": {}, "annotations": {}}}
        for i in xrange(annotations_size):
            statistics["counts"]["annotations"][str(i)] = {}
            statistics["probabilities"]["annotations"][str(i)] = {}
        nucleotides_length = 0
        kmers_length = 0

        for sequence in sequences:

            nucleotides = cs.Counter([j[0] for j in sequence[1]])
            nucleotides_length += len(sequence[1])
            for key in nucleotides.keys():
                if key in statistics["counts"]["nucleotides"].keys():
                    statistics["counts"]["nucleotides"][key] += nucleotides[key]
                else:
                    statistics["counts"]["nucleotides"][key] = nucleotides[key]

            for i in xrange(annotations_size):
                annotations = cs.Counter([j[i + 1] for j in sequence[1]])
                for key in annotations.keys():
                    if key in statistics["counts"]["annotations"][str(i)].keys():
                        statistics["counts"]["annotations"][str(i)][key] += annotations[key]
                    else:
                        statistics["counts"]["annotations"][str(i)][key] = annotations[key]

            kmers = cs.Counter([''.join([j[0] for j in sequence[1][i: i + kmer_size]]) for i in range(len(sequence[1]) - kmer_size + 1)])
            kmers_length += (len(sequence[1]) - kmer_size + 1)
            for key in kmers.keys():
                if key in statistics["counts"]["kmers"].keys():
                    statistics["counts"]["kmers"][key] += kmers[key]
                else:
                    statistics["counts"]["kmers"][key] = kmers[key]

        for key in statistics["counts"]["nucleotides"].keys():
            statistics["probabilities"]["nucleotides"][key] = float(statistics["counts"]["nucleotides"][key]) / float(nucleotides_length)

        for key in statistics["counts"]["kmers"].keys():
            statistics["probabilities"]["kmers"][key] = float(statistics["counts"]["kmers"][key]) / float(kmers_length)

        for i in xrange(annotations_size):
            for key in statistics["counts"]["annotations"][str(i)].keys():
                statistics["probabilities"]["annotations"][str(i)][key] = float(statistics["counts"]["annotations"][str(i)][key]) / float(nucleotides_length)

        if dump:
            with open("nucleotides_and_kmers_counts_" + file_name + "_" + str(kmer_size) + '.pkl', 'wb') as f:
                pickle.dump(statistics, f)

        return statistics

    @staticmethod
    def check_for_emissions(emissions_alphabet_train, emissions_alphabet_test, extracted_variables, final_generated_test_sequences):
        """
        Method that changes emissions that are only in test data to the generic emission or emission next to it.

        :param emissions_alphabet_train: List of emissions in training data.
        :param emissions_alphabet_test: List of emissions in test data.
        :param extracted_variables: Variables extracted for sequence and HMM building.
        :param final_generated_test_sequences: List of test sequences to correct.
        :returns: List of corrected test sequences.
        """

        emissions_only_in_test = list(set(emissions_alphabet_test) - set(emissions_alphabet_train))

        release_sequences = []

        if len(emissions_only_in_test) > 0:

            if len(extracted_variables[0][0]) > 1:

                for emission in emissions_only_in_test:

                    new_emission = "X" * len(extracted_variables[0][0]) + emission[len(extracted_variables[0][0]): len(emission)]

                    if new_emission in emissions_alphabet_train:

                        for sequence in final_generated_test_sequences:
                            if emission in sequence[1]:
                                for i in xrange(len(sequence[1])):
                                    if sequence[1][i] == emission:
                                        sequence[1][i] = new_emission
                    else:

                        seq_index = 0
                        for sequence in final_generated_test_sequences:
                            if emission in sequence[1]:
                                for i in xrange(len(sequence[1])):
                                    if sequence[1][i] == emission:
                                        if i > 0:
                                            sequence[1][i] = sequence[1][i - 1]
                                        else:
                                            i_index = 1
                                            while sequence[1][i + i_index] == sequence[1][i]:
                                                if i_index == len(sequence[1]) - 1:
                                                    release_sequences.append(seq_index)
                                                    break
                                                i_index += 1
                                            sequence[1][i] = sequence[1][i + i_index]
                            seq_index += 1
            else:
                for emission in emissions_only_in_test:

                    seq_index = 0
                    for sequence in final_generated_test_sequences:
                        if emission in sequence[1]:
                            for i in xrange(len(sequence[1])):
                                if sequence[1][i] == emission:

                                    if i > 0:
                                        sequence[1][i] = sequence[1][i - 1]
                                    else:
                                        i_index = 1
                                        while sequence[1][i + i_index] == sequence[1][i]:
                                            if i_index == len(sequence[1]) - 1:
                                                release_sequences.append(seq_index)
                                                break
                                            i_index += 1
                                        sequence[1][i] = sequence[1][i + i_index]
                        seq_index += 1

        if len(release_sequences) > 0:
            for rs in release_sequences:
                print "Removed: ", rs

            print "release_sequences", release_sequences
            print len(final_generated_test_sequences)
            print final_generated_test_sequences
            final_generated_test_sequences = np.delete(final_generated_test_sequences, np.array(release_sequences), axis=0)
            print "---------------------"
            print final_generated_test_sequences
            print len(final_generated_test_sequences)
        return final_generated_test_sequences

    def visualize_probability_of_prediction(self, file_manager, sequences, hmm, experiment, file_name, state_name):
        """
        Method that visualizes list of points(probability of crosslink, iclip intensity) for given list of sequences.

        :param file_manager: Given file manager object.
        :param sequences: Given list of sequences.
        :param hmm: Given HMM.
        :param experiment: Given ID of single experiment.
        :param file_name: Given name of output file.
        :param state_name: Given name of state.
        """

        points = []
        for s in sequences:
            p = self.probability_of_prediction(s[0], s[1], file_manager, experiment, hmm.get_state_by_name(state_name)[1])
            points.extend(p)

        points = np.array(points)

        self.draw_linera_regression(points, 'Probability of prediction versus intensity', 'Probabilities', 'Intensities', "probability_of_prediction_" + file_name)

    @staticmethod
    def probability_of_prediction(sequence, y_score, file_manager, experiment, state_index):
        """
        Method that makes list of points(probability of crosslink, iclip intensity) for given sequence.

        :param sequence: Given sequence.
        :param y_score: Matrix of probabilities.
        :param file_manager: Given file manager object.
        :param experiment: Given ID of single experiment.
        :param state_index: Given ID of wanted state inside the matrix of probabilities.
        :returns: List of points(probability of crosslink, iclip intensity).
        """

        gene = sequence[2][1]
        iclip = pickle.load(gzip.open(file_manager.iclip_files_path + gene['files']['iclip']))
        gene_iclip_raw = iclip[gene['gene_location'][0]: gene['gene_location'][1]]
        gene_iclip = iclip[gene['gene_location'][0]: gene['gene_location'][1]].nonzero()

        points = []
        for i in xrange(len(gene_iclip[0])):
            if gene_iclip[1][i] == int(experiment) and gene_iclip[0][i] < y_score.shape[1]:
                points.append([y_score[state_index][gene_iclip[0][i]], gene_iclip_raw[gene_iclip[0][i], gene_iclip[1][i]], ])

        return points

    def visualize_sequence(self, sequence, file_manager, hmm, iclip_experiment, visualization_type):
        """
        Method that visualizes single or ali iclip experiment values and annotations for single sequence.

        :param sequence: Given sequence.
        :param file_manager: Given file manager object.
        :param hmm: Given HMM.
        :param iclip_experiment: Given ID of single experiment.
        :param visualization_type: String that decides the visualization of single or all iclip experiments values.
        """

        if visualization_type == "single":

            gene = sequence[2][1]

            predicted_sequence, y_score = hmm.posterior_decoding_scaled(sequence[1])
            predicted_sequence = predicted_sequence[1::].astype(int)
            y_score = y_score[:,1::]

            #print gene['files']['iclip']
            iclip = pickle.load(gzip.open(file_manager.iclip_files_path + gene['files']['iclip']))
            gene_iclip = iclip[gene['gene_location'][0]: gene['gene_location'][1]][:, iclip_experiment]

            annotations = pickle.load(gzip.open(file_manager.annot_files_path + gene['files']['annot']))
            gene_annotations = annotations[gene['gene_location'][0]: gene['gene_location'][1]].astype(int)

            length = predicted_sequence.shape[0]
            gene_iclip = gene_iclip[0:length]
            gene_annotations = np.rot90(gene_annotations[0:length, :])

            #print y_score[1,:]
            self.draw_sequence_annotations_predictions_iclip(sequence, iclip_experiment, gene_annotations, predicted_sequence, gene_iclip, length, 'Intensities and annotations for a single sequence and single iCLIP experiment')
            #self.draw_sequence_annotations_predictions_iclip(sequence, iclip_experiment, gene_annotations, predicted_sequence, gene_iclip, length, 'Prikaz dejanskih in napovedanih CLIP vrednosti in vrednost ostalih atributov za posamezno zaporedje in eksperiment')

    def visualize_new_sequence(self, sequence, file_manager, hmm, iclip_experiment, state):
        """
        Method that visualizes single or ali iclip experiment values and annotations for single sequence.

        :param sequence: Given sequence.
        :param file_manager: Given file manager object.
        :param hmm: Given HMM.
        :param iclip_experiment: Given ID of single experiment.
        :param visualization_type: String that decides the visualization of single or all iclip experiments values.
        """

        gene = sequence[2].tolist()

        predicted_sequence, y_score = hmm.posterior_decoding_scaled(sequence[1])
        predicted_sequence = predicted_sequence[1::].astype(int)
        prediction_probabilities = {}
        for s in state:
            s_id = hmm.get_state_by_name(s)[1]
            prediction_probabilities[s] = y_score[s_id,1::]

        (iclip_a, _), iclip_labels = pickle.load(gzip.open(file_manager.iclip_files_path + gene['iclip_file']))
        gene_iclip = iclip_a[:, iclip_experiment]
        current_gene_iclip = gene_iclip.toarray().reshape(gene_iclip.toarray().shape[0])

        region_a, region_labels = pickle.load(gzip.open(file_manager.annot_files_path + gene['region_file']))
        current_region = region_a.astype(int)

        rnafold_a = pickle.load(gzip.open(file_manager.rnafold_files_path + gene['rnafold_file']))
        #print rnafold_a
        #current_rnafold = rnafold_a[:, rnafold_a.shape[1]/2].astype(bool).astype(int)
        current_rnafold = rnafold_a.astype(int)

        length = predicted_sequence.shape[0]
        current_gene_iclip = current_gene_iclip[0:length]
        current_region = current_region[:, 0:length]
        current_rnafold = current_rnafold[0:length]

        direction = [s for s in re.split("_|-", gene['iclip_file'])][4]

        if direction == "":
            current_rnafold = current_rnafold[::-1]
            current_region = current_region[::-1]
            current_gene_iclip = current_gene_iclip[::-1]

        self.draw_new_sequence_annotations_predictions_iclip(sequence, iclip_experiment, current_region, current_rnafold, predicted_sequence, current_gene_iclip, prediction_probabilities, length, 'Prikaz dejanskih in napovedanih CLIP vrednosti in vrednost ostalih atributov')
        """

        #print gene['files']['iclip']
        iclip = pickle.load(gzip.open(file_manager.iclip_files_path + gene['files']['iclip']))
        gene_iclip = iclip[gene['gene_location'][0]: gene['gene_location'][1]][:, iclip_experiment]

        annotations = pickle.load(gzip.open(file_manager.annot_files_path + gene['files']['annot']))
        gene_annotations = annotations[gene['gene_location'][0]: gene['gene_location'][1]].astype(int)

        length = predicted_sequence.shape[0]
        gene_iclip = gene_iclip[0:length]
        gene_annotations = np.rot90(gene_annotations[0:length, :])

        #print y_score[1,:]
        self.draw_sequence_annotations_predictions_iclip(sequence, iclip_experiment, gene_annotations, predicted_sequence, gene_iclip, length, 'Intensities and annotations for a single sequence and single iCLIP experiment')
        """

    def reduce_sequences(self, sequences, state_min_ratio, sequence_max_length, dump, file_name):
        """
        Method that discards sequences that have crosslink ratio lower then given threshold.

        :param sequences: List of given sequences.
        :param state_min_ratio: Which states represent crosslinks and given threshold of minimal ratio, for discarding sequences.
        :param dump: Logical switch for writing results to file.
        :param file_name: String that represent the name of output file.
        :returns: List of sequences that have crosslink ratio atleat the same as given minimal ratio.
        """

        ratios = {}
        for key in state_min_ratio:
            ratios[key] = {}
        new_sequence = []
        index = 0
        for sequence in sequences:
            if len(sequence[0]) < sequence_max_length:
                counts = cs.Counter(sequence[0])
                #print counts
                check = []
                #print "------------------------------------------------------------------------------------"
                for state_check in state_min_ratio:
                    #print state_check
                    ratios[state_check][index] = float(counts[state_check]) / float(len(sequence[0]))
                    #print ratios[state_check]
                    #print state_min_ratio[state_check]
                    if ratios[state_check][index] > state_min_ratio[state_check]:
                        #print "Check"
                        check.append(1)
                #if len(check) == len(state_min_ratio.keys()):
                if len(check) > 0:
                    #print "Added"
                    new_sequence.append(sequence)
                index += 1

        for key in state_min_ratio:
            if dump:
                sorted_list = sorted(ratios[key].items(), key=lambda x: x[1], reverse=True)

                with open("visualize_sequence_state_ratio_data_" + key + "_" + file_name + '.pkl', 'wb') as f:
                    pickle.dump(sorted_list, f)

                self.draw_plot(range(len(sorted_list)), [x[1] for x in sorted_list], 'State ratio ' + key  + " (" + str(len(new_sequence)) + ")", "Sequence", "Ratio", "visualize_sequence_state_ratio_" + key + "_" + file_name)

        return np.array(new_sequence)
    
    @staticmethod
    def matthews_correlation_coefficient(y_true, y_predicted):
        """
        Method that calculated Matthew's correlation coefficient for given sequence of true states and for given sequence of predicted states.

        :param y_true: Given list of true states.
        :param y_predicted: Given list of predicted states.
        :returns: Number that represents Matthew's correlation coefficient.
        """

        return mt.matthews_corrcoef(y_true, y_predicted)

    def rank_experiment_predictions(self, list_of_hmms, chosen_experiment, sequences, chosen_state, dump, iclip_info_file, evaluation):
        """
        Method that evaluates and ranks HMMs predictions for given iCLIP experiment.

        :param list_of_hmms: List of given HMMs and their parameters.
        :param chosen_experiment: Id of chosen experiment.
        :param sequences: List of test sequences.
        :param chosen_state: String that represents the chosen state for evaluation.
        :param dump: Logical switch for writing results to file.
        :param iclip_info_file: String that represents the name of file, which represents all iCLIP experiment data.
        :param evaluation: String that represents the evaluation method.
        """

        file_manager = FileManager()
        file_manager.iclip_info_file = iclip_info_file
        chosen_experiment_final_generated_sequences = pickle.load(gzip.open(list_of_hmms[chosen_experiment][0]))
        chosen_experiment_sequences = np.append(chosen_experiment_final_generated_sequences[0], chosen_experiment_final_generated_sequences[1], axis=0)
        chosen_experiment_generic_sequences = pickle.load(gzip.open(list_of_hmms[chosen_experiment][2]))

        results = []

        for i in list_of_hmms:
        #for i in [27]:
            print "Experiment:", i
            temp_experiment_sequences = []
            temp_experiment_final_generated_sequences = []
            if i != chosen_experiment:
                temp_experiment_final_generated_sequences = pickle.load(gzip.open(list_of_hmms[i][0]))
                temp_experiment_sequences = np.append(temp_experiment_final_generated_sequences[0], temp_experiment_final_generated_sequences[1], axis=0)
            else:
                temp_experiment_final_generated_sequences = chosen_experiment_final_generated_sequences
                temp_experiment_sequences = chosen_experiment_sequences

            temp_hmm = pickle.load(gzip.open(list_of_hmms[i][1]))
            if isinstance(temp_hmm, list):
                temp_hmm = temp_hmm[0]

            scores = np.array([])
            tests = np.array([])
            for sequence in sequences:

                info = sequence[2].tolist()

                temp_sequence = [x for x in temp_experiment_sequences if x[2].tolist()['indexes'] == sequence[2].tolist()['indexes']]
                if not temp_sequence:
                    temp_sequence = [x for x in chosen_experiment_generic_sequences if x[2].tolist()['indexes'] == sequence[2].tolist()['indexes']][0]
                    temp_sequence, emissions_alphabet_test, states_alphabet_test = file_manager.make_sequences([temp_sequence], len(temp_experiment_final_generated_sequences[4][0][0]), temp_experiment_final_generated_sequences[4][0], temp_experiment_final_generated_sequences[4][1])
                    temp_sequence = self.check_for_emissions([x[0].name for x in temp_hmm.emission_alphabet], emissions_alphabet_test, temp_experiment_final_generated_sequences[4], temp_sequence)

                if len(temp_sequence) > 0:
                    temp_sequence = temp_sequence[0]

                    predicted_sequence, y_score = temp_hmm.posterior_decoding_scaled(temp_sequence[1])
                    y_test = temp_hmm.create_state_matrix(temp_sequence)

                    if scores.shape[0] == 0:
                        scores = y_score
                    else:
                        scores = np.concatenate((scores, y_score), axis=1)

                    if tests.shape[0] == 0:
                        tests = y_test
                    else:
                        tests = np.concatenate((tests, y_test), axis=1)

            if evaluation == "ROC":
                n_classes = len(temp_hmm.state_alphabet)
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for j in xrange(n_classes):
                    fpr[temp_hmm.get_state_by_index(j)[0].name], tpr[temp_hmm.get_state_by_index(j)[0].name], _ = mt.roc_curve(tests[j, :], scores[j, :])
                    roc_auc[temp_hmm.get_state_by_index(j)[0].name] = mt.auc(fpr[temp_hmm.get_state_by_index(j)[0].name], tpr[temp_hmm.get_state_by_index(j)[0].name])

                results.append((i, roc_auc[chosen_state]))

            elif evaluation == "MAE":
                n_classes = len(temp_hmm.states_alphabet)
                mae = dict()
                for j in xrange(n_classes):
                    mae[temp_hmm.get_state_by_index(j)[0].name] = 1.0 - mt.mean_absolute_error(tests[j, :], scores[j, :])

                results.append((i, mae[chosen_state]))

        sorted_list = sorted(results, key=lambda x: x[1], reverse=True)

        if dump:
            self.draw_bar_chart(sorted_list, 'AUC of experiments', "Experiment id", "AUC", "rank_of_preditions_" + str(chosen_experiment))
            data = np.array(sorted_list)
            colLabels = ["Experiment ID", "AUC"]
            labels = pickle.load(open(iclip_info_file))
            rowLabels = [labels[x[0]] for x in sorted_list]
            title = "Experiment " + str(chosen_experiment)
            self.draw_table(data, colLabels, rowLabels, title, "rank_of_preditions_table_" + str(chosen_experiment))
            with open("rank_of_preditions_sorted_list_" + str(chosen_experiment) + '.pkl', 'wb') as f:
                pickle.dump(sorted_list, f)

        return sorted_list

    def rank_experiment_from_hmms_build_with_predictions(self, train_sequences, test_sequences, list_of_hmms, chosen_experiment_id, chosen_state, file_name, dump, evaluation):
        """
        Method that evaluates and ranks HMMs predictions for given iCLIP experiment.

        :param train_sequences: List of train sequences.
        :param test_sequences: List of test sequences.
        :param list_of_hmms: List of given HMMs and their parameters.
        :param chosen_experiment_id: Id of chosen experiment.
        :param chosen_state: String that represents the chosen state for evaluation.
        :param file_name: Name of the output file.
        :param dump: Logical switch for writing results to file.
        :param evaluation: String that represents the evaluation method.
        """

        hmm_type = "discrete"
        file_manager = FileManager()
        chosen_experiment_final_generated_sequences = pickle.load(open(list_of_hmms[chosen_experiment_id][0]))
        chosen_experiment_sequences = np.append(chosen_experiment_final_generated_sequences[0], chosen_experiment_final_generated_sequences[1], axis=0)
        chosen_experiment_generic_sequences = pickle.load(open(list_of_hmms[chosen_experiment_id][2]))

        results = []

        for i in xrange(len(list_of_hmms)):
            print "Experiment:", i
            temp_experiment_sequences = []
            temp_experiment_final_generated_sequences = []
            if i != chosen_experiment_id:
                temp_experiment_final_generated_sequences = pickle.load(open(list_of_hmms[i][0]))
                temp_experiment_sequences = np.append(temp_experiment_final_generated_sequences[0], temp_experiment_final_generated_sequences[1], axis=0)
            else:
                temp_experiment_final_generated_sequences = chosen_experiment_final_generated_sequences
                temp_experiment_sequences = chosen_experiment_sequences
            temp_hmm = pickle.load(open(list_of_hmms[i][1]))

            new_train_sequences = []
            train_emissions = []
            train_states = []
            for sequence in train_sequences:

                temp_sequence = [x for x in temp_experiment_sequences if x[2][0] == sequence[2][0]]
                if not temp_sequence:
                    temp_sequence = [x for x in chosen_experiment_generic_sequences if x[2][0] == sequence[2][0]][0]
                    temp_sequence, emissions_alphabet_test, states_alphabet_test = file_manager.make_sequences([temp_sequence], len(temp_experiment_final_generated_sequences[4][0][0]), temp_experiment_final_generated_sequences[4][0], temp_experiment_final_generated_sequences[4][1])
                    temp_sequence = self.check_for_emissions([x[0].name for x in temp_hmm.emission_alphabet], emissions_alphabet_test, temp_experiment_final_generated_sequences[4], temp_sequence)

                temp_sequence = temp_sequence[0]

                predicted_sequence, y_score = temp_hmm.posterior_decoding_scaled(temp_sequence[1])
                predicted_sequence = predicted_sequence[1::]

                for state in temp_hmm.state_alphabet:
                    train_states.append(state[0].name)
                for emission in cs.Counter(predicted_sequence).keys():
                    train_emissions.append(emission)

                new_train_sequences.append(np.array([sequence[0], np.array(predicted_sequence), sequence[2]]))
            train_emissions = list(set(train_emissions))
            train_states = list(set(train_states))

            new_test_sequences = []
            test_emissions = []
            test_states = []
            for sequence in train_sequences:

                temp_sequence = [x for x in temp_experiment_sequences if x[2][0] == sequence[2][0]]
                if not temp_sequence:
                    temp_sequence = [x for x in chosen_experiment_generic_sequences if x[2][0] == sequence[2][0]][0]
                    temp_sequence, emissions_alphabet_test, states_alphabet_test = file_manager.make_sequences([temp_sequence], len(temp_experiment_final_generated_sequences[4][0][0]), temp_experiment_final_generated_sequences[4][0], temp_experiment_final_generated_sequences[4][1])
                    temp_sequence = self.check_for_emissions([x[0].name for x in temp_hmm.emission_alphabet], emissions_alphabet_test, temp_experiment_final_generated_sequences[4], temp_sequence)

                temp_sequence = temp_sequence[0]

                predicted_sequence, y_score = temp_hmm.posterior_decoding_scaled(temp_sequence[1])
                predicted_sequence = predicted_sequence[1::]

                for state in temp_hmm.state_alphabet:
                    test_states.append(state[0].name)
                for emission in cs.Counter(predicted_sequence).keys():
                    test_emissions.append(emission)

                new_test_sequences.append(np.array([sequence[0], np.array(predicted_sequence), sequence[2]]))
            test_emissions = list(set(test_emissions))
            test_states = list(set(test_states))

            result = self.cross_validation_of_hmm(new_train_sequences, new_test_sequences, train_states, train_emissions, hmm_type, file_name + "_inner_predicted_hmm", dump, evaluation)
            results.append((list_of_hmms[i][3], result[0][chosen_state]))

        draw_sorted_list = sorted(results, key=lambda x: x[1], reverse=True)
        self.draw_bar_chart(draw_sorted_list, 'AUC of experiments', "Experiment id", evaluation, "rank_experiment_from_hmms_build_with_predictions_" + str(chosen_experiment_id))

    def visualize_reduced_sequences(self, generic_sequences_folder, filename_starts_with, state_min_ratio, dump, sequence_max_length):
        """
        Method that visualizes the number of non zero sequences per experiment.

        :param generic_sequences_folder: Path to the generic sequences.
        :param filename_starts_with: The start of files with generic sequences.
        :param state_min_ratio: Minimum probabilityy of iCLIP intensities.
        :param dump: Logical switch for writing results to file.
        :param sequence_max_length: Maximum length of single sequence.
        """

        len_seq = []
        len_nuc = []
        per_seq = []
        per_nuc = []
        data = {}
        for filename in os.listdir(generic_sequences_folder):
            if filename.startswith(filename_starts_with):
                print filename
                sequences = pickle.load(gzip.open(generic_sequences_folder + filename))
                print len(sequences), sum([len(x[0]) for x in sequences])
                reduced_sequences = self.reduce_sequences(sequences, state_min_ratio, sequence_max_length, dump, filename[len(filename_starts_with):len(filename) - 4])
                experiment_name = filename[len(filename_starts_with):len(filename) - 4]
                len_seq.append(len(reduced_sequences))
                len_nuc.append(sum([len(x[0]) for x in reduced_sequences]))
                per_seq.append(float(len(reduced_sequences)) / float(len(sequences)))
                per_nuc.append(float(sum([len(x[0]) for x in reduced_sequences])) / float(sum([len(x[0]) for x in sequences])))
                print len(reduced_sequences), sum([len(x[0]) for x in reduced_sequences]), float(len(reduced_sequences)) / float(len(sequences)), float(sum([len(x[0]) for x in reduced_sequences])) / float(sum([len(x[0]) for x in sequences]))
                data[[int(s) for s in experiment_name.split("_") if s.isdigit()][0]] = len(reduced_sequences)

        print "len_seq", float(sum(len_seq)) / float(len(len_seq))
        print "len_nuc", float(sum(len_nuc)) / float(len(len_nuc))
        print "per_seq", float(sum(per_seq)) / float(len(per_seq))
        print "per_nuc", float(sum(per_nuc)) / float(len(per_nuc))

        draw_sorted_list = sorted(data.items(), key=lambda x: x[1], reverse=True)
        self.draw_bar_chart(draw_sorted_list, 'Number of nonzero sequences', "Experiment id", "Number of sequences", "histogram_of_nonzero_sequences_per_experiment")

    def find_best_sequences(self, generic_sequences_folder, filename_starts_with, state_min_ratio, dump, experiment_ids):
        """
        Method that finds the sequences that have intensities in as many experiments as possible.

        :param generic_sequences_folder: Path to the generic sequences.
        :param filename_starts_with: The start of files with generic sequences.
        :param state_min_ratio: Minimum probabilityy of iCLIP intensities.
        :param dump: Logical switch for writing results to file.
        :param experiment_ids: List of experiment ids.
        :returns: Dictionary of sequences and experiment ids.
        """

        data = {}
        for filename in os.listdir(generic_sequences_folder):
            if filename.startswith(filename_starts_with):
                experiment_name = filename[len(filename_starts_with):len(filename) - 4]
                experiment_id = [int(s) for s in experiment_name.split("_") if s.isdigit()][0]
                if experiment_id in experiment_ids:
                    print filename
                    sequences = pickle.load(open(generic_sequences_folder + filename))
                    reduced_sequences = self.reduce_sequences(sequences, state_min_ratio, dump, filename[len(filename_starts_with):len(filename) - 4])

                    for sequence in reduced_sequences:

                        info = sequence[2].tolist()
                        s_index = (info["indexes"][0], info["indexes"][1], info["indexes"][2])

                        if s_index in data.keys():
                            data[s_index].append(experiment_id)
                        else:
                            data[s_index] = [experiment_id]

        print data
        print len(data.keys())

        if dump:
            visualize_data = {}
            for key in data.keys():
                n_experiments = len(data[key])
                if n_experiments in visualize_data.keys():
                    visualize_data[n_experiments] += 1
                else:
                    visualize_data[n_experiments] = 1

            print visualize_data
            print len(visualize_data.keys())

            new_visualize_data = {}
            for key in visualize_data.keys():
                temp_sum = 0
                for inner_key in xrange(key, max(visualize_data.keys()) + 1):
                    temp_sum += visualize_data[inner_key]
                new_visualize_data[key] = temp_sum

            draw_sorted_list = sorted(new_visualize_data.items(), key=lambda x: x[1], reverse=True)
            self.draw_bar_chart(draw_sorted_list, 'Number of nonzero sequences for all experiments', "Number of experiments", "Number of sequences", "find_best_sequences_" + str(len(experiment_ids)))

            with open("find_best_sequences.pkl", 'wb') as f:
                pickle.dump(data, f)

        return data

    @staticmethod
    def create_network_of_experiments_by_predictions(rank_files_path, rank_file_startswith, iclip_info_file, top_treshold_predictions):
        """
        Method that builds graph from ranks how HMMs of other experiments predict single experiment.

        :param rank_files_path: Path to rank files.
        :param rank_file_startswith: The start of rank files.
        :param iclip_info_file: Path to file with data about iCLIP experiments.
        :param top_treshold_predictions: How many the best predictions we take.
        """

        groups = {
            1:  "A",
            2:  "A",
            3:  "A",
            4:  "A",
            5:  "A",
            6:  "B",
            7:  "B",
            8:  "C",
            9:  "C",
            10:  "C",
            11:  "C",
            12:  "D",
            13:  "E",
            14:  "E",
            15:  "F",
            16:  "G",
            17:  "G",
            18:  "H",
            19:  "H",
            20:  "H",
            21:  "I",
            22:  "J",
            23:  "S",
            24:  "K",
            25:  "L",
            26:  "M",
            27:  "N",
            28:  "O",
            29:  "P",
            30:  "P",
            31:  "R",
            32:  "R",
        }

        iclip_experiments = {
            0: 7,
            1: 29,
            2: 26,
            3: 8,
            4: 15,
            5: 23,
            6: 18,
            7: 16,
            8: 2,
            9: 30,
            10: 1,
            11: 9,
            12: 14,
            13: 22,
            14: 12,
            15: 19,
            16: 32,
            17: 24,
            18: 5,
            19: 20,
            20: 31,
            21: 17,
            22: 13,
            23: 21,
            24: 10,
            25: 6,
            26: 3,
            27: 25,
            28: 4,
            29: 27,
            30: 28,
            31: 11}

        nodes = set()
        nodes_and_edges = dict()

        for filename in os.listdir(rank_files_path):
            if filename.startswith(rank_file_startswith):
                experiment_id = [int(s) for s in re.split("_|-|\.", filename) if s.isdigit()][0]
                list_of_predictions = pickle.load(open(rank_files_path + filename))
                prediction = [x[1] for x in list_of_predictions if x[0] == experiment_id][0]
                edges = [(str(iclip_experiments[x[0]]), str(iclip_experiments[experiment_id]), x[1]) for x in list_of_predictions if x[1] > prediction]

                final_edges = []
                for edge in edges:
                    if groups[int(edge[0])] != groups[int(edge[1])]:
                        final_edges.append(edge)

                nodes.add(str(iclip_experiments[experiment_id]))
                for i in [x[0] for x in edges]:
                    nodes.add(i)
                nodes_and_edges[str(iclip_experiments[experiment_id])] = {"edges": final_edges[0:top_treshold_predictions]}


        n = 1.0 / len(nodes)

        color = 0
        for i in nodes:
            if i in nodes_and_edges.keys():
                nodes_and_edges[i]["color"] = color
            else:
                nodes_and_edges[i] = {}
                nodes_and_edges[i]["color"] = color
            color += n

        experiment_labels = pickle.load(open(iclip_info_file))

        G = nx.Graph()

        test_edges = []
        for key in nodes_and_edges:
            if "edges" in nodes_and_edges[key].keys():
                for edge in nodes_and_edges[key]["edges"]:
                    test_edges.append(edge[2])

        labels = dict()
        for key in nodes_and_edges:
            G.add_node(int(key))

            labels[int(key)] = int(key)

            if "edges" in nodes_and_edges[key].keys():
                for edge in nodes_and_edges[key]["edges"]:
                    #print str(edge[0]), str(edge[1]), edge[2], int(round(((edge[2] - min(test_edges)) * ((10 - 1)) / (max(test_edges) - min(test_edges))) + 1))
                    G.add_edge(int(edge[0]), int(edge[1]), weight=int(round(((edge[2] - min(test_edges)) * ((6 - 1)) / (max(test_edges) - min(test_edges))) + 1)))

        edge_labels = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])

        print edge_labels

        """
        print test_edges
        print max(test_edges)
        print min(test_edges)
        test_edges2 = []
        for i in test_edges:
            test_edges2.append(int(round(((i - min(test_edges)) * ((10 - 1)) / (max(test_edges) - min(test_edges))) + 1)))

        print test_edges2
        """

        """
        weights = []
        for key in nodes_and_edges:
            if "edges" in nodes_and_edges[key].keys():
                ind = 0
                for edge in nodes_and_edges[key]["edges"]:
                    if ind < top_treshold_predictions:
                        weights.append(float(edge[2]))
                        ind += 1

        for key in nodes_and_edges:
            if "edges" in nodes_and_edges[key].keys():
                ind = 0
                for edge in nodes_and_edges[key]["edges"]:
                    if ind < top_treshold_predictions:
                        G.add_edge(str(edge[0]), str(edge[1]), weight=((float(1.0 - edge[2]) - min(weights)) * (10 - 1)) / (max(weights) - min(weights)) + 1)
                        ind += 1
        """

        f = plt.figure(1)
        ax = f.add_subplot(1,1,1)
        for key in [str(y) for y in sorted([int(x) for x in nodes_and_edges.keys()])]:
            key_s = [x for x in iclip_experiments.items() if x[1] == int(key)][0][0]
            label = ""
            if key_s in [25, 0]:
                label = experiment_labels[int(key_s)].split(".")[0].split("_")[0].split("-")[2] + experiment_labels[int(key_s)].split(".")[0].split("_")[1]

            else:

                label = experiment_labels[int(key_s)].split(".")[0].split("_")[1]

                if label[len(label)-4:len(label)] == "NASE":
                    label = label[:len(label)-4]

            labels[int(key)] = str(key + "\n" + label)
            ax.plot([0], [0], color=hsv_to_rgb([nodes_and_edges[key]["color"], 1.0, 1.0]), label=str(key + " "*(3-len(key)) + label))

        node_color = [hsv_to_rgb([nodes_and_edges[str(v)]["color"], 1.0, 1.0]) for v in G]
        edge_color = [hsv_to_rgb([nodes_and_edges[str(x[0])]["color"], 1.0, 1.0]) for x in G.edges()]

        pos = nx.spring_layout(G, iterations=5000, k=0.20)
        #nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=node_size, node_color=node_color, linewidths=0.3, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, with_labels=False, node_size=300, node_color=hsv_to_rgb([0.0, 0.0, 1.0]), linewidths=0.0, alpha=1)
        nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=300, node_color="w", linewidths=0.3, alpha=0.5)
        #nx.draw_networkx_edges(G, pos, arrows=True, width=0.3, alpha=0.5)
        nx.draw_networkx_edges(G, pos, arrows=True, width=0.2, alpha=0.5)
        #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=3, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels, font_size=3)

        plt.axis('off')
        #plt.legend(loc=1, prop={'size': 5})
        plt.savefig("create_network_of_experiments_by_predictions_" + str(top_treshold_predictions) + ".pdf")

    @staticmethod
    def sequences_split_ids(percent_of_train_sequences, path_to_single_generic_sequence, dump):
        """
        Method that splits sequences with given threshold.

        :param percent_of_train_sequences: Percent of train sequences.
        :param path_to_single_generic_sequence: Path to generic sequences file.
        :param dump: Logical switch for writing results to file.
        :returns: List of train and test sequences.
        """

        sequences = pickle.load(open(path_to_single_generic_sequence))
        number_of_train_sequences = int(len(sequences) * percent_of_train_sequences / 100)
        train_ids = rd.sample(range(len(sequences)), number_of_train_sequences)
        test_ids = [p for p in range(len(sequences)) if p not in train_ids]

        if dump:
            with open('sequences_split_ids.pkl', 'wb') as f:
                pickle.dump([train_ids, test_ids], f)

        return [train_ids, test_ids]

    def distributions_of_new_data_zero_nucleotides(self, data, n_sequences):
        """
        Method that draws the lengths until first nucleotide without annotation, and first nucleotide without annotation and iCLIP intensity higher than zero.

        :param data: Dictionary of positions.
        :param n_sequences: Number of sequences.
        """

        for key in data:
            bins = dict()
            for seq in data[key]["first_region_zero_and_iclip_nonzero_dict"]:
                value = int(seq[4] * 100)
                if value in bins.keys():
                    bins[value] += 1
                else:
                    bins[value] = 1
            bins[100] = n_sequences - len(data[key]["first_region_zero_and_iclip_nonzero_dict"])
            self.draw_plot(bins.keys(), bins.values(), "Distribution of first region zero and iclip nonzero for experiment: " + key, "Percent", "Count", "distribution_region_zero_and_iclip_nonzero_" + key)

        for key in data:
            bins = dict()
            for seq in data[key]["first_region_zero_dict"]:
                value = int(seq[4] * 100)
                if value in bins.keys():
                    bins[value] += 1
                else:
                    bins[value] = 1
            bins[100] = n_sequences - len(data[key]["first_region_zero_dict"])
            self.draw_plot(bins.keys(), bins.values(), "Distribution of first region zero for experiment: " + key, "Percent", "Count", "distribution_region_zero_" + key)

    def length_of_sequences(self, sequences):
        """
        Method that draws graph of sequences lengths, for better representation they are represented with intervals.

        :param sequences: List of sequences.
        """

        lengths = dict()
        for s in sequences:
            if len(s[0]) in lengths.keys():
                lengths[len(s[0])] += 1
            else:
                lengths[len(s[0])] = 1

        max_length = max(lengths.keys())
        bins = dict()
        for key in lengths:
            value = int(float(key)/float(max_length) * 100)
            if value in bins.keys():
                bins[value] += lengths[key]
            else:
                bins[value] = lengths[key]

        self.draw_plot(bins.keys(), bins.values(), u"Dolžina zaporedij (maksimalna dolžina: " + str(max_length) + u")", u"Dolžina [%]", u"Število zaporedij", "length_of_sequences")

    @staticmethod
    def draw_bar_chart(tuple_list, title, xlabel, ylabel, save_file_name):
        """
        Method that draws bar chart.

        :param tuple_list: List of (x, y) tuples.
        :param title: String that represents the title of the graph.
        :param xlabel: String that represents label for x axis.
        :param ylabel: String that represents label for y axis.
        :param save_file_name: Output file name.
        """

        width = 0.4
        font_size = 8
        n_elements = len(tuple_list)
        x = np.arange(1, n_elements + 1)
        y = [num for (s, num) in tuple_list]
        labels = [s for (s, num) in tuple_list]

        plt.figure()

        ax = plt.axes()
        ax.yaxis.grid()

        #plt.grid()
        plt.bar(x, y, width, color='r')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x + width/2.0, labels, fontsize=font_size, rotation=0)
        #plt.xticks(x + width/2.0, labels, fontsize=font_size)
        plt.xlim([min(x) - 0.25, max(x) + 0.75])
        plt.savefig(save_file_name+ '.pdf', bbox_inches='tight')
        plt.close()

    def draw_bar_chart_with_table(self, tuple_list, title, xlabel, ylabel, save_file_name, selected_experiments, iclip_labels, experiment_id, state, min_in_state_ratio):
        """
        Method that draws bar chart with table.

        :param tuple_list: List of (x, y) tuples.
        :param title: String that represents the title of the graph.
        :param xlabel: String that represents label for x axis.
        :param ylabel: String that represents label for y axis.
        :param save_file_name: Output file name.
        :param selected_experiments: List of selected experiments.
        :param iclip_labels: List of iCLIP experiments labels.
        :param experiment_id: The chosen experiment id.
        :param state: The chosen state.
        :param min_in_state_ratio: The minimal probability of single emission inside state.
        """

        width = 0.75
        font_size = 8
        n_elements = len(tuple_list)
        x = np.arange(1, n_elements + 1)
        y = [num for (s, num) in tuple_list]
        labels = [s for (s, num) in tuple_list]

        plt.figure()
        plt.grid()
        plt.bar(x, y, width, color='r')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x + width/2.0, labels, fontsize=font_size, rotation=90)
        plt.xlim([min(x) - 0.25, max(x) + 1])
        plt.savefig(save_file_name + '.pdf', bbox_inches='tight')
        plt.close()

        colLabels = []
        for i in selected_experiments:
            colLabels.append(str(i) + " " + iclip_labels[i][0:15])

        colLabels.append("Ratio")

        rowLabels = [x[0] for x in tuple_list]

        title = "Ratio of emissions for state " + state + " in experiment " + str(experiment_id) + " " + iclip_labels[experiment_id][0:15] + " min_in_state_ratio: " + str(min_in_state_ratio)

        data = []
        for i in tuple_list:
            line = []
            for j in i[0]:
                line.append(j)
            line.append(i[1])
            data.append(line)

        self.draw_table(data, colLabels, rowLabels, title, save_file_name + "_table")

    @staticmethod
    def draw_roc_curve(fpr, tpr, roc_auc, hmm, title, xlabel, ylabel, save_file_name):
        """
        Method that ROC curve.

        :param fpr: Dictionary of false positive rates.
        :param tpr: Dictionary of true positive rates.
        :param roc_auc: Dictionary of AUC values.
        :param hmm: Given chosen HMM.
        :param title: String that represents the title of the graph.
        :param xlabel: String that represents label for x axis.
        :param ylabel: String that represents label for y axis.
        :param save_file_name: Output file name.
        """
        n_classes = len(hmm.state_alphabet)
        plt.figure()
        plt.grid()
        for i in range(n_classes):
            plt.plot(fpr[hmm.get_state_by_index(i)[0].name], tpr[hmm.get_state_by_index(i)[0].name], label='ROC curve of class {0} (area = {1:0.2f})'.format(hmm.get_state_by_index(i)[0].name, roc_auc[hmm.get_state_by_index(i)[0].name]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(save_file_name + '.pdf', bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_sequence_annotations_predictions_iclip(sequence, iclip_experiment, gene_annotations, predicted_sequence, gene_iclip, length, title):
        """
        Method that draws true and predicted iCLIP values and annotations.

        :param sequence: Given sequence.
        :param iclip_experiment: Chosen iCLIP experiment.
        :param gene_annotations: Given list annotations.
        :param predicted_sequence: Given predicted sequence.
        :param gene_iclip: Given iCLIP values.
        :param length: Given length of sequence.
        :param title: Given title of the graph.
        """

        f, axarr = plt.subplots(7, sharex=True)
        f.suptitle(title)

        axarr[0].plot(range(length), gene_iclip)
        axarr[0].set_ylim([0.0, max(gene_iclip) + 0.05])
        axarr[0].set_xlim([0.0, length + 0.05])
        axarr[0].set_ylabel("iCLIP")

        axarr[1].plot(range(length), predicted_sequence)
        axarr[1].set_ylim([0.0, max(predicted_sequence) + 0.05])
        axarr[1].set_xlim([0.0, length + 0.05])
        axarr[1].set_ylabel("Prediction")

        annotations = ["Exon", "Intron", "5UTR", "CDS", "3UTR"]
        for i in xrange(2, 7):
            axarr[i].plot(range(length), gene_annotations[i - 2, :])
            axarr[i].set_ylim([0.0, 1.05])
            axarr[i].set_xlim([0.0, length + 0.05])
            axarr[i].set_ylabel(annotations[i - 2])

        plt.savefig("visualization_of_sequence_" + iclip_experiment + "_" + sequence[2][0][0] + "_" + sequence[2][0][1] + "_"  + sequence[2][0][2] + '.pdf', bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_new_sequence_annotations_predictions_iclip(sequence, iclip_experiment, gene_annotations, gene_rnafold, predicted_sequence, gene_iclip, prediction_probabilities, length, title):
        """
        Method that draws true and predicted iCLIP values, annotations and RNAFold.

        :param sequence: Given sequence.
        :param iclip_experiment: Chosen iCLIP experiment.
        :param gene_annotations: Given list annotations.
        :param gene_rnafold: Given RNAFold values.
        :param predicted_sequence: Given predicted sequence.
        :param gene_iclip: Given iCLIP values.
        :param prediction_probabilities: Given prediction probabilities.
        :param length: Given length of sequence.
        :param title: Given title of the graph.
        """

        font_size = 6
        l_offset = length * 0.05
        e_name = sequence[2].tolist()['experiment'][1]
        s_name = sequence[2].tolist()['iclip_file']
        s_name = s_name[0:len(s_name) - 7]

        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)

        f, axarr = plt.subplots(10, sharex=True)
        f.suptitle(title)

        axarr[0].plot(range(length), gene_iclip)
        offset = float(max(gene_iclip)) * 0.1
        axarr[0].set_ylim([0.0 - offset, max(gene_iclip) + offset])
        axarr[0].set_xlim([0.0 - l_offset, length + l_offset])
        axarr[0].set_ylabel("CLIP", fontsize=font_size)
        #axarr[0].set_title("ICLIP_Experiment: " + str(iclip_experiment) + " " + e_name + "\n Sequence: " + s_name, fontsize=font_size)
        axarr[0].set_title("Eksperiment: hnRNPC (2)\n Zaporedje: " + s_name.split("_")[3][4:], fontsize=font_size)

        max_all = []
        for s in prediction_probabilities:
            max_all.append(max(prediction_probabilities[s]))
            axarr[1].plot(range(length), prediction_probabilities[s], label=s)
        offset = max(max_all) * 0.1
        axarr[1].set_ylim([0.0 - offset, max(max_all) + offset])
        axarr[1].set_xlim([0.0 - l_offset, length + l_offset])
        axarr[1].set_ylabel("P(N)", fontsize=font_size)
        #axarr[1].legend(loc="lower right", prop={'size':4})

        max_all = []
        for s in prediction_probabilities:
            for i in xrange(len(prediction_probabilities[s])):
                if prediction_probabilities[s][i] < 1.0:
                    prediction_probabilities[s][i] = math.fabs(math.log(1.0 - prediction_probabilities[s][i]))
                else:
                    prediction_probabilities[s][i] = 40.0
            max_all.append(max(prediction_probabilities[s]))
            axarr[2].plot(range(length), prediction_probabilities[s], label=s)
        offset = max(max_all) * 0.1
        axarr[2].set_ylim([0.0 - offset, max(max_all) + offset])
        axarr[2].set_xlim([0.0 - l_offset, length + l_offset])
        axarr[2].set_ylabel("|log(1-P(N))|", fontsize=font_size)
        #axarr[2].legend(loc="lower right", prop={'size':4})

        annotations = ["Exon", "Intron", "5UTR", "ORF", "3UTR", "ncRNA"]

        for i in xrange(3, 9):
            axarr[i].plot(range(length), gene_annotations[i - 3, :])
            offset = max(gene_annotations[i - 3, :]) * 0.1
            axarr[i].set_ylim([0.0 - offset, max(gene_annotations[i - 3, :]) + offset])
            axarr[i].set_xlim([0.0 - l_offset, length + l_offset])
            axarr[i].set_ylabel(annotations[i - 3], fontsize=font_size)

        axarr[9].plot(range(length), gene_rnafold)
        offset = max(gene_rnafold) * 0.1
        axarr[9].set_ylim([0.0 - offset, max(gene_rnafold) + offset])
        axarr[9].set_xlim([0.0 - l_offset, length + l_offset])
        axarr[9].set_ylabel("RNA fold", fontsize=font_size)

        plt.savefig("visualization_of_sequence_of_experiment_" + str(iclip_experiment) + "_" + s_name + '.pdf', bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_linera_regression(points, title, xlabel, ylabel, file_name):
        """
        Method that draws linear regression for given points.

        :param points: 2D List of points.
        :param title: String that represents the title of the graph.
        :param xlabel: String that represents label for x axis.
        :param ylabel: String that represents label for y axis.
        :param save_file_name: Output file name.
        """

        points[:, 0] /= max(points[:, 0])
        points[:, 1] /= max(points[:, 1])

        slope, intercept, r_value, p_value, std_err = sps.linregress(points[:, 0], points[:, 1])

        plt.figure()
        plt.grid()
        line = slope * points[:, 0] + intercept
        plt.plot(points[:, 0], line, 'r-', points[:, 0], points[:, 1], '.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(file_name + '.pdf', bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_plot(xdata, ydata, title, xlabel, ylabel, file_name):
        """
        Method that draws plot.

        :param xdata: List of x axis values.
        :param ydata: List of y axis values.
        :param title: String that represents the title of the graph.
        :param xlabel: String that represents label for x axis.
        :param ylabel: String that represents label for y axis.
        :param save_file_name: Output file name.
        """

        plt.figure()
        plt.plot(xdata, ydata)
        #plt.plot([0, 2042], [0.00001, 0.00001], 'r-')
        plt.grid(True)
        plt.xlim([0 - max(xdata)/100, max(xdata) + max(xdata)/100])
        plt.ylim([0 - max(ydata)/100, max(ydata) + max(ydata)/100])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(file_name + '.pdf', bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_heatmap(k_means, number_of_clusters, counts, data, header, title, xlabel, ylabel, file_name):
        """
        Method that draws heatmap.

        :param k_means: Dictionary of k-means clustering.
        :param number_of_clusters: Number of clusters.
        :param counts: List of counts
        :param data: List of values.
        :param header: Given header.
        :param title: String that represents the title of the graph.
        :param xlabel: String that represents label for x axis.
        :param ylabel: String that represents label for y axis.
        :param save_file_name: Output file name.
        """

        plt.figure()
        plt.imshow(k_means[0], interpolation='none')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.yticks(range(number_of_clusters), [str(i) + ": " + str(counts[i]) for i in range(number_of_clusters)])
        plt.xlabel(xlabel)
        plt.xticks(range(len(data[0])), header, fontsize=4, rotation=90)
        plt.bone()
        plt.colorbar()
        plt.savefig(file_name + '.pdf', bbox_inches='tight')
        plt.close()

    def draw_table(self, data, colLabels, rowLabels, title, file_name):
        """
        Method that draws table.

        :param data: List of values.
        :param colLabels: Columns label.
        :param rowLabels: Rows label.
        :param title: String that represents the title of the graph.
        :param file_name: Output file name.
        """

        nrows, ncols = len(data) + 1, len(colLabels)
        hcell, wcell = 0.3, 1.
        hpad, wpad = 0, 0
        fig = plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
        ax = fig.add_subplot(111)
        ax.axis('off')

        plt.title(title, fontsize=8, loc='right')
        the_table = ax.table(cellText=data, colLabels=colLabels, rowLabels=rowLabels, cellLoc='left', rowLoc='left', colLoc='left', loc='left')
        plt.savefig(file_name + '.pdf', bbox_inches='tight')

    def draw_table2(self, data, experiment, rowLabels, title, file_name):
        """
        Method that draws table.

        :param data: List of values.
        :param experiment: Chosen experiment.
        :param rowLabels: Rows label.
        :param title: String that represents the title of the graph.
        :param file_name: Output file name.
        """

        rows = TableFactory.RowSpec(TableFactory.ColumnSpec('name', 'Name', width=5), TableFactory.ColumnSpec('e_number', 'Experiment number', width=1), TableFactory.ColumnSpec('auc', 'AUC', width=1))

        lines = []
        for i in data:
            lines.append([rows({'name': rowLabels[i[0]].split(".")[0][0:60], 'e_number': i[0], 'auc': "{0:.3f}".format(round(i[1], 3))})])

        outfile = open(file_name + '.pdf', 'wb')

        outfile.write(TableFactory.PDFTable(title, rowLabels[experiment].split(".")[0], headers=rows).render(lines))