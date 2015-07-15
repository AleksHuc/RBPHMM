import pickle
import gzip
import os
import re
import numpy
import glob


class FileManager:
    """
    Class that represent file manager.
    """

    def __init__(self):
        self.region_files_path = ""
        self.genes_files_path = ""
        self.iclip_files_path = ""
        self.rnafold_files_path = ""
        self.iclip_file_name = ""
        self.region_file_name = ""
        self.rnafold_file_name = ""
        self.sequence_file_name = ""
        self.gene_file = ""
        self.region_file = ""
        self.string_file = ""
        self.iclip_info_file = ""
        self.ignored_keys_list = ""
        self.sequence_files_path = ""
        self.files = {}
        self.rev_code = {'A': 'T', 'T': 'A', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

    def make_data(self):
        """
        Method that makes initial data packages.
        """

        chr = 'chr1'

        data = {}
        with open(self.gene_file) as f:
            for line in f:

                substrings = line.split()

                if substrings[3] == chr and substrings[2] == "protein_coding":
                    substrings.remove(substrings[7])
                    substrings.remove(substrings[7])
                    substrings[5] = int(substrings[5][1:len(substrings[5]) - 1])
                    substrings[6] = int(substrings[6][0:len(substrings[6]) - 1])

                    key = substrings[0]
                    if key not in data.keys():
                        data[key] = dict()
                        filename = '_chrome-' + chr + '_strand-' + substrings[4] + '_gid-' + key + '_start-' + str(substrings[5]) + '_end-' + str(substrings[6]) + '.pkl'
                        substrings.append(filename)
                        data[key]["gene"] = substrings

        with open(self.region_file) as f:
            for line in f:

                substrings = line.split()
                if len(substrings) > 7 and substrings[3] == chr:

                    if len(substrings) == 9:
                        if substrings[7] in data.keys():
                            data[substrings[7]][substrings[1]] = substrings
                    else:
                        if substrings[6] in data.keys():
                            data[substrings[6]][substrings[1]] = substrings

        key_bindings = {'exon': 0, 'intron': 1, '5UTR': 2, 'ORF': 3, '3UTR': 4, 'ncRNA': 5}

        for key in data.keys():

            gene = data[key]["gene"]
            gene_length = gene[6] - gene[5] + 1
            region = numpy.zeros(shape=(6, gene_length), dtype=bool)

            dna_sequence = self.read_dna_string_from_to(gene[5], gene[6] + 1, gene[4])
            dna_sequence_list = numpy.array(list(dna_sequence))
            dna_sequence_list = dna_sequence_list[0: dna_sequence_list.shape[0] - 1]
            output_file = gzip.GzipFile('SEQUENCE' + gene[7], 'wb')
            output_file.write(pickle.dumps(dna_sequence_list, - 1))
            output_file.close()

            for key2 in data[key].keys():
                if key2 != "gene":
                    intervals = [[y - gene[5] for y in map(int, x.split("-"))] for x in data[key][key2][5].split(",")]
                    for interval in intervals:
                        region[key_bindings[key2], interval[0]:interval[1] + 1] = True
                        if key2 in ['5UTR', 'ORF', '3UTR']:
                            region[key_bindings["exon"], interval[0]:interval[1] + 1] = True

            region = region[:, 0: region.shape[1] - 1]

            for i in xrange(region.shape[1]):
                if not any(region[:, i]):
                    region[:, i][5] = True

            """
            f, axarr = plt.subplots(6, sharex=True)
            f.suptitle(str(gene[0]) + "_" + str(gene[1]) + "_" + str(gene[2]) + "_" + str(gene[3]) + "_" + str(gene[4]) + "_" + str(gene[5]) + "_" + str(gene[6]))

            annotations = ["Exon", "Intron", "5UTR", "ORF", "3UTR", "ncRNA"]
            for i in xrange(len(annotations)):
                axarr[i].plot(range(region.shape[1]), region[i, :])
                axarr[i].set_ylim([0.0, 1.05])
                axarr[i].set_xlim([0.0, region.shape[1] + 0.05])
                axarr[i].set_ylabel(annotations[i])

            plt.show()
            """

            output_file = gzip.GzipFile('REGION' + gene[7], 'wb')
            output_file.write(pickle.dumps((region, key_bindings), -1))
            output_file.close()

    def read_dna_string_from_to(self, start_location, stop_location, string_direction):
        """
        Method that returns the chosen substring of the whole nucleotide string.

        - A - Adenine
        - C - Cytosine
        - G - Guanine
        - T - Thymine
        - N - Any of the nucleotides.

        :param start_location: Index that represents the start of substring.
        :param stop_location: Index that represents the end of substring.
        :param string_direction: String ('+' or '-') that sets the direction in which the nucleotides are read.
        :returns: The chosen substring of the whole nucleotide string.
        """

        string_file = open(self.string_file)
        dna_string = string_file.read()[start_location: stop_location]

        if string_direction == "-":
            return self.reverse_complement(dna_string)
        elif string_direction == "+":
            return dna_string
        else:
            print "ERROR: String direction is wrong!"
            return ""

    def reverse_complement(self, s):
        """
        Method that returns the reverse complement of a given DNA sequence.

        :param s: Input DNA sequence.
        :returns: Reverse complement in the correct orientation.
        """

        return "".join([self.rev_code[n] for n in s[::-1]])

    @staticmethod
    def make_sequences(sequences, kmer_size, kmers, annotations_list):
        """
        Method that generates specific sequences from a general ones.

        :param sequences: Given general sequences.
        :param kmer_size: Given size of kmers.
        :param kmers: List of kmers to use in emissions.
        :param annotations_list: List of annotations which to use in emissions.
        :returns: New specific sequences and new state and emission alphabets.
        """

        new_sequences = []
        emissions_alphabet = set()
        state_alphabet = []

        for sequence in sequences:

            if len(sequence[0]) >= kmer_size:

                states = sequence[0][0: (len(sequence[0]) - kmer_size + 1)]
                state_alphabet.extend(list(set(states)))

                emissions = []
                for i in xrange(len(sequence[0]) - kmer_size + 1):

                    emission = ''.join([j[0] for j in sequence[1][i: i + kmer_size]])
                    if emission not in kmers:
                        emission = "X" * kmer_size

                    for j in annotations_list:
                        emission += sequence[1][i][j + 1]

                    emissions.append(emission)
                    emissions_alphabet.add(emission)

                emissions = numpy.array(emissions)
                states = numpy.array(states)

                new_sequences.append([states, emissions, sequence[2]])

        new_sequences = numpy.array(new_sequences)

        return new_sequences, list(emissions_alphabet), list(set(state_alphabet))

    @staticmethod
    def second_stage_make_sequences(sequences, chosen_emissions):
        """
        Method that makes second stage sequences.

        :param sequences: List of sequences.
        :param chosen_emissions: List of chosen emissions.
        :returns: New sequences and list of emissions and states.
        """

        new_sequences = []
        emissions_alphabet = set()
        state_alphabet = []

        for sequence in sequences:
            states = sequence[0]
            state_alphabet.extend(list(set(states)))

            emissions = []
            for e in sequence[1]:
                if e not in chosen_emissions:
                    emission = "X" * len(e)
                else:
                    emission = e

                emissions.append(emission)
                emissions_alphabet.add(emission)

            emissions = numpy.array(emissions)
            states = numpy.array(states)

            new_sequences.append([states, emissions, sequence[2]])

        new_sequences = numpy.array(new_sequences)

        return new_sequences, list(emissions_alphabet), list(set(state_alphabet))

    def make_sequences_from_predictions(self, sequences, list_of_hmms, selected_experiments, chosen_experiment_id, analytics_manager, file_name, dump):
        """
        Method that makes sequences from predictions.

        :param sequences: List of sequences.
        :param list_of_hmms: List of HMMS.
        :param selected_experiments: List of selected experiments.
        :param chosen_experiment_id: Id of chosen experiment.
        :param analytics_manager: Analytics Manager object.
        :param file_name: Output file name.
        """

        chosen_experiment_generic_sequences = pickle.load(gzip.open(list_of_hmms[chosen_experiment_id][2]))

        dictionary_of_sequences_predictions = {}
        for i in selected_experiments:
            print "HMM", i
            temp_experiment_final_generated_sequences = pickle.load(gzip.open(list_of_hmms[i][0]))
            temp_experiment_sequences = numpy.append(temp_experiment_final_generated_sequences[0], temp_experiment_final_generated_sequences[1], axis=0)

            temp_hmm = pickle.load(gzip.open(list_of_hmms[i][1]))
            if isinstance(temp_hmm, list):
                temp_hmm = temp_hmm[0]

            for sequence in sequences:

                temp_sequence = [x for x in temp_experiment_sequences if x[2].tolist()['indexes'] == sequence[2].tolist()['indexes']]
                if not temp_sequence:
                    temp_sequence = [x for x in chosen_experiment_generic_sequences if x[2].tolist()['indexes'] == sequence[2].tolist()['indexes']][0]
                    temp_sequence, emissions_alphabet_test, states_alphabet_test = self.make_sequences([temp_sequence], len(temp_experiment_final_generated_sequences[4][0][0]), temp_experiment_final_generated_sequences[4][0], temp_experiment_final_generated_sequences[4][1])
                    temp_sequence = analytics_manager.check_for_emissions([x[0].name for x in temp_hmm.emission_alphabet], emissions_alphabet_test, temp_experiment_final_generated_sequences[4], temp_sequence)

                temp_sequence = temp_sequence[0]

                temp_predicted_sequence, _ = temp_hmm.posterior_decoding_scaled(temp_sequence[1])
                temp_predicted_sequence = temp_predicted_sequence[1::]

                #print sequence[2].tolist()
                s_key = (sequence[2].tolist()['indexes'][0], sequence[2].tolist()['indexes'][1])
                if s_key in dictionary_of_sequences_predictions.keys():
                    dictionary_of_sequences_predictions[s_key][i] = temp_predicted_sequence
                else:
                    dictionary_of_sequences_predictions[s_key] = {}
                    dictionary_of_sequences_predictions[s_key][i] = temp_predicted_sequence

        #print "selected_experiments", selected_experiments

        for c_experiments in range(2, len(selected_experiments) + 1):
            #print "c_experiments", c_experiments
            new_sequences = []
            index = 0
            for sequence in sequences:

                s_key = (sequence[2].tolist()['indexes'][0], sequence[2].tolist()['indexes'][1])
                all_sequence_predictions = dictionary_of_sequences_predictions[s_key]

                #print "all_sequence_predictions", all_sequence_predictions

                #print "all_sequence_predictions.values()", all_sequence_predictions.values()

                sequence_predictions = []

                for j in selected_experiments[0:c_experiments]:
                    sequence_predictions.append(all_sequence_predictions[j])

                #print "sequence_predictions", sequence_predictions

                length = min([len(x) for x in sequence_predictions])

                emission_sequence = []
                for i in xrange(length):
                    emission = [x[i] for x in sequence_predictions]
                    emission_sequence.append("".join(emission))

                if len(sequence[0][0:length]) != len(emission_sequence):
                    print "len(sequence[0][0:length])", len(sequence[0][0:length])
                    print "len(emission_sequence)", len(emission_sequence)

                new_sequences.append(numpy.array([sequence[0][0:length], numpy.array(emission_sequence), sequence[2]]))
                index += 1

            if dump:
                output_file = gzip.GzipFile(file_name + "_" + str(c_experiments) + '.pkl', 'wb')
                output_file.write(pickle.dumps(numpy.array(new_sequences), -1))
                output_file.close()

    def make_sequences_from_iclip(self, sequences, list_of_hmms, selected_experiments, chosen_experiment_id, file_name, dump):
        """
        Method that makes sequences from iCLIP data.

        :param sequences: List of sequences.
        :param list_of_hmms: List of HMMS.
        :param selected_experiments: List of selected experiments.
        :param chosen_experiment_id: Id of chosen experiment.
        :param file_name: Output file name.
        :param dump: Logical switch for writing results to file.
        :returns: List of new sequences.
        """

        dictionary_of_sequences_predictions = {}
        for i in selected_experiments:
            print "HMM", i
            temp_experiment_generic_sequences = pickle.load(open(list_of_hmms[i][2]))

            for sequence in sequences:

                temp_sequence = [x for x in temp_experiment_generic_sequences if x[2].tolist()['indexes'] == sequence[2].tolist()['indexes']][0]

                temp_iclip_sequence = temp_sequence[0]
                s_key = (sequence[2].tolist()['indexes'][0], sequence[2].tolist()['indexes'][1], sequence[2].tolist()['indexes'][2])
                if s_key in dictionary_of_sequences_predictions.keys():
                    dictionary_of_sequences_predictions[s_key][i] = temp_iclip_sequence
                else:
                    dictionary_of_sequences_predictions[s_key] = {}
                    dictionary_of_sequences_predictions[s_key][i] = temp_iclip_sequence

        new_sequences = []
        index = 0
        for sequence in sequences:
            #print "Sequence: ", index
            s_key = (sequence[2].tolist()['indexes'][0], sequence[2].tolist()['indexes'][1], sequence[2].tolist()['indexes'][2])
            sequence_predictions = dictionary_of_sequences_predictions[s_key]

            #length = min([len(x) for x in sequence_predictions.values()])
            length = len(sequence[0])
            emission_sequence = []
            for i in xrange(length):
                emission = [x[i] for x in sequence_predictions.values()]
                emission_sequence.append("".join(emission))

            if len(sequence[0][0:length]) != len(emission_sequence):
                print "len(sequence[0][0:length])", len(sequence[0][0:length])
                print "len(emission_sequence)", len(emission_sequence)

            new_sequences.append(numpy.array([sequence[0][0:length], numpy.array(emission_sequence), sequence[2]]))
            index += 1

        if dump:
            output_file = gzip.GzipFile('second_stage_iclip_sequences_' + str(chosen_experiment_id) + "_" + file_name + '.pkl', 'wb')
            output_file.write(pickle.dumps(numpy.array(new_sequences), -1))
            output_file.close()

        return numpy.array(new_sequences)

    def make_generic_sequences(self, states_dict, order, list_of_experiments):
        """
        Method that generates generic sequences of iclip states and emissions from nucleotides and annotations.

        :param states_dict: Dictionary of states and their values.
        :param order: Given order of positive states.
        :param list_of_experiments: Given list of experiments for which to make sequences.
        """

        for experiment in list_of_experiments:

            print "Experiment: " + str(experiment) + ": "

            sequences = []
            index = 0

            for fi, f in enumerate(glob.glob(self.iclip_files_path + self.iclip_file_name)):

                if index % 100 == 0:
                    print index

                index += 1

                (iclip_a, _), iclip_labels = pickle.load(gzip.open(f))
                experiment_name = iclip_labels[experiment]

                direction = [s for s in re.split("_|-", f)][4]
                if direction == "":
                    direction = "-"

                indexes = [int(s) for s in re.split("_|-|\.", f) if s.isdigit()]

                current_string_file_name = self.sequence_file_name[0:len([s for s in re.split("_", self.sequence_file_name)][0])] + f[len(self.iclip_files_path) + len([s for s in re.split("_", self.iclip_file_name)][0]):len(f)]
                string = pickle.load(gzip.open(self.sequence_files_path + current_string_file_name))

                iclip = iclip_a[:, experiment].toarray()

                states = numpy.array(["0"] * len(string))

                for i in iclip.nonzero()[0]:
                    value = iclip[i][0]
                    value_state = "0"
                    for s in states_dict:
                        if states_dict[s][0] < value <= states_dict[s][1]:
                            value_state = s
                    states[i] = value_state

                    for j in xrange(1, order):
                        if (i + j) < (len(states) - 1):
                            states[i + j] = value_state
                        if (i - j) > 0:
                            states[i - j] = value_state

                current_region_file_name = self.region_file_name[0:len([s for s in re.split("_", self.region_file_name)][0])] + f[len(self.iclip_files_path) + len([s for s in re.split("_", self.iclip_file_name)][0]):len(f)]
                region_a, region_labels = pickle.load(gzip.open(self.region_files_path + current_region_file_name))
                current_region = region_a.astype(int)

                current_rnafold_file_name = self.rnafold_file_name[0:len([s for s in re.split("_", self.rnafold_file_name)][0])] + f[len(self.iclip_files_path) + len([s for s in re.split("_", self.iclip_file_name)][0]):len(f)]
                rnafold_a = pickle.load(gzip.open(self.rnafold_files_path + current_rnafold_file_name))
                current_rnafold = rnafold_a.astype(int)

                if direction == "-":
                    current_rnafold = current_rnafold[::-1]
                    current_region = current_region[::-1]
                    states = states[::-1]

                emissions = []
                for index_s in xrange(string.shape[0]):
                    emission = string[index_s]
                    emission += ''.join([str(x) for x in current_region[:, index_s]])
                    emission += str(current_rnafold[index_s])
                    emissions.append(emission)

                additional_data = {"indexes": indexes, "iclip_file": f[len(self.iclip_files_path): len(f)], "region_file": current_region_file_name, "rnafold_file": current_rnafold_file_name, "direction": direction, "experiment": [experiment, experiment_name]}

                sequences.append(numpy.array([numpy.array(states), numpy.array(emissions), numpy.array(additional_data)]))

            output_file = gzip.GzipFile("make_generic_sequences_from_new_data_" + str(experiment) + "_" + str(order) + '.pkl', 'wb')
            output_file.write(pickle.dumps(sequences, -1))
            output_file.close()

    def check_new_data_iclip(self, list_of_experiments):
        """
        Method that counts the values if iCLIP intensities.

        :param list_of_experiments: List of experiments.
        :returns: List of results.
        """

        iclip_file_name = "CLIP_chrome-chr1_strand*.pkl.gz"

        data = dict()

        for experiment in list_of_experiments:

            data[str(experiment)] = dict()
            data[str(experiment)]["counts"] = dict()

            length = 0
            index = 0

            for fi, f in enumerate(glob.glob(self.iclip_files_path + iclip_file_name)):

                if index % 100 == 0:
                    print index

                index += 1

                (iclip_a, _), iclip_labels = pickle.load(gzip.open(f))
                data[str(experiment)]["name"] = iclip_labels[experiment]

                iclip = iclip_a[:, experiment].toarray()

                for i in xrange(len(iclip)):

                    current_iclip = iclip[i]

                    if current_iclip[0] > 0:
                        if current_iclip[0] in data[str(experiment)]["counts"].keys():
                            data[str(experiment)]["counts"][current_iclip[0]] += 1
                        else:
                            data[str(experiment)]["counts"][current_iclip[0]] = 1

            data[str(experiment)]["length"] = length

        with open("check_new_data_iclip" + str(list_of_experiments[0]) + '.pkl', 'wb') as new_file:
                pickle.dump(data, new_file)

        return data

    @staticmethod
    def make_hmm_list(final_sequences_path, final_sequences_startswith, hmm_path, hmm_startswith, hmm_endswith, generic_sequences_path, generic_sequences_startswith, rank_of_predictions_path, rank_of_predictions_startswith):
        """
        Method that makes list of HMMs and other data.

        :param final_sequences_path: Path to final sequences.
        :param final_sequences_startswith: Start of final sequences file.
        :param hmm_path: Path to hmm file.
        :param hmm_startswith: Start of hmm file.
        :param hmm_endswith: End of hmm file
        :param generic_sequences_path: Path to generic sequences.
        :param generic_sequences_startswith: Start of generic sequences file.
        :param rank_of_predictions_path: Path to rank files.
        :param rank_of_predictions_startswith: Start of rank file.
        :returns. Dictionary of data.
        """


        list_of_hmms = {}

        for file_name in os.listdir(final_sequences_path):
            if file_name.startswith(final_sequences_startswith):
                experiment_id = [int(s) for s in re.split("_|-|\.", file_name) if s.isdigit()][0]
                list_of_hmms[experiment_id] = [final_sequences_path + "\\" + file_name]

        for file_name in os.listdir(hmm_path):
            if file_name.startswith(hmm_startswith) and file_name.endswith(hmm_endswith + ".pkl"):
                experiment_id = [int(s) for s in re.split("_|-|\.", file_name) if s.isdigit()][0]
                list_of_hmms[experiment_id].append(hmm_path + file_name)

        for file_name in os.listdir(generic_sequences_path):
            if file_name.startswith(generic_sequences_startswith):
                experiment_id = [int(s) for s in re.split("_|-|\.", file_name) if s.isdigit()][0]
                list_of_hmms[experiment_id].append(generic_sequences_path + file_name)

        for file_name in os.listdir(rank_of_predictions_path):
            if file_name.startswith(rank_of_predictions_startswith):
                experiment_id = [int(s) for s in re.split("_|-|\.", file_name) if s.isdigit()][0]
                list_of_hmms[experiment_id].append(rank_of_predictions_path + file_name)

        return list_of_hmms

    @staticmethod
    def read_chosen_experiemnt(file_path):
        """
        Method that read data about experiment.

        :param file_path: Path to file.
        :returns: Dictionary of results.
        """

        chosen_experiemnts = dict()
        e = "Experiment:  "
        h = "HMM "
        with open(file_path, "r") as ins:
            current = ""
            cntr = 0
            for line in ins:
                if line.startswith(e):
                    current = line.split()[1]
                    chosen_experiemnts[current] = []
                    cntr = 0
                elif line.startswith(h) and cntr < 5:
                    chosen_experiemnts[current].append(int(line.split()[1]))
                    cntr += 1

        return chosen_experiemnts