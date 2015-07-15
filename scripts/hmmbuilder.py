# -*- coding: utf-8 -*-

import time
import datetime
import pickle
import gzip
import os
import glob
import numpy
import collections
import random
import math
import sklearn.metrics

from iCLIPHMM.core import AnalyticsManager
from iCLIPHMM.core import FileManager
from iCLIPHMM.core import HiddenMarkovModel


def main():
    """
    Method for testing and evaluating of HMMs.
    """

    print "#"*99 + " START " + "#"*99
    start = time.time()

    """
    file_path = "E:\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\\"

    array_of_sums = [0] * 2004

    for file_name in os.listdir(file_path):
        if file_name.startswith("visualize_sequence_state_ratio_data_"):
            opened_file = pickle.load(open(file_path + file_name))
            print opened_file
            for i in xrange(len(opened_file)):
                array_of_sums[i] += opened_file[i][1]

    d_array_of_sums = [0] * 2004
    for i in xrange(len(array_of_sums)):
        d_array_of_sums[i] = float(array_of_sums[i]) / float(32)

    am = AnalyticsManager()
    am.draw_plot(range(2004), d_array_of_sums, u"Povprečje verjetnosti CLIP intenzitet v vseh zaporedjih", u"Število zaporedij, ki predstavljajo gene", "Verjetnosti CLIP intenzitet", "probability_of_intensities_2_")

    #seq = pickle.load(gzip.open("C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\make_sequences_from_predictions_train_28_4_2.pkl"))
    #print seq

    #seq = pickle.load(gzip.open("C:\Aleks\First stage HMMs Final\\final_generated_sequences_28_iter-3.pkl"))
    #print len(seq[0])
    """
    """
    seq = pickle.load(open("C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\iclip_labels.pkl"))

    dictionary = dict()
    index = 0
    for i in seq:
        dictionary[index] = [i]
        index += 1

    print dictionary
    """
    """
    seq = pickle.load(gzip.open("D:\Dokumenti\FRI\Magistrska\Results\First stage HMMs Final\hmm_7_iter-3_final.pkl"))
    c_hmm = seq[0]

    c_hmm.print_matrix(c_hmm.emission_matrix)
    c_hmm.print_matrix(c_hmm.transition_matrix)
    """
    ############################################################################################
    #  Statistics  #############################################################################
    ############################################################################################

    combined_models = {
        1: {2: ['11', '10'], 3: ['101', '111', '100', '110'], 4: ['0101', '1011', '1111', '0111', '0001', '1001', '0011', '1101', '1010', '1000'], 5: ['01011', '10111', '10110', '11111', '01111', '11110', '00011', '00111', '10011', '00010']},
        2: {2: ['10', '11'], 3: ['011', '001', '101', '111'], 4: ['0111', '0011', '1011', '0110', '0010', '1111', '1010', '1110'], 5: ['01111', '10101', '00111', '00101', '01101', '10111', '00100', '11111', '11101']},
        3: {2: ['11', '10'], 3: ['111', '101', '100', '110'], 4: ['1110', '1111', '1010', '1000', '1101', '1100'], 5: ['11010', '11000', '11101', '11111', '10000', '11100', '10100']},
        4: {2: ['11', '10'], 3: ['111', '100', '101'], 4: ['1111', '1110', '1000', '1011', '1010', '0011'], 5: ['11111', '11100', '11101', '10000', '01101']},
        5: {2: ['11', '10'], 3: ['101', '111', '110', '100'], 4: ['1011', '0111', '1111', '0001', '1101', '1001', '0101', '0010', '1010'], 5: ['01111', '10111', '11111', '01101', '01001', '00011', '01011', '10101', '00101', '00001']},
        6: {2: ['11', '10'], 3: ['111', '110', '101', '001', '011'], 4: ['1110', '0001', '1111', '1100', '1010', '1000', '1101', '0010', '0101', '1011'], 5: ['11100', '00010', '11110', '11000', '10100', '10000', '11010', '00100', '01010', '10110']},
        7: {2: ['11', '01', '10'], 3: ['111', '101', '110', '010', '011'], 4: ['1110', '0001', '1111', '1100', '1000', '1010', '0100', '0011', '1011', '1101'], 5: ['00011', '11101', '00001', '00101', '00010', '11111', '10000', '01001', '11001', '10100']},
        8: {2: ['10', '11'], 3: ['111', '101', '011', '100', '110'], 4: ['1110', '1111', '1001', '1011', '1010', '0111', '1101', '0110', '1000'], 5: ['11101', '10101', '11111', '10111', '10011', '01111', '01100', '11110']},
        9: {2: ['11', '10', '01'], 3: ['111', '100', '110', '001', '101', '010'], 4: ['1111', '1110', '1000', '1100', '1011', '0010', '1001', '0011', '1101'], 5: ['10001', '10011', '11000', '11111', '10111', '10101', '11101']},
        10: {2: ['11', '10'], 3: ['111', '101', '100', '011', '001', '110'], 4: ['1111', '1101', '1001', '1011', '0111', '0101', '0011', '1010', '1110', '1000'], 5: ['11110', '11111', '11010', '10110', '10010', '01110', '10001']},
        11: {2: ['11', '10'], 3: ['110', '111', '100'], 4: ['1101', '0111', '1111', '0001', '0101', '1100', '1001', '1011'], 5: ['01110', '11011', '01111', '10110', '11110', '11010', '10010', '11001', '11111', '00010']},
        12: {2: ['11', '10'], 3: ['111', '101', '100', '011'], 4: ['1001', '1101', '1110', '0110', '1010', '1111', '1011', '0010', '1000'], 5: ['10010', '10011', '11010', '11101', '11111', '11100', '00101', '10111']},
        13: {2: ['10', '11'], 3: ['101', '110', '100', '111', '001'], 4: ['1011', '1111', '1101', '1001', '1010', '1000', '1100', '0011', '1110'], 5: ['10111', '11111', '11011', '10110', '10011', '11000', '10101', '10000', '10010', '10001']},
        14: {2: ['11', '10'], 3: ['111', '101', '011', '001', '100', '110'], 4: ['1111', '1110', '1011', '1010', '0110', '0111', '0011', '0010', '1001'], 5: ['11111', '11100', '10111', '11101', '10101', '11110', '01100', '10100', '10001', '00100']},
        15: {2: ['10', '11'], 3: ['111', '101', '100', '110'], 4: ['1111', '1011', '0101', '0001', '1010', '1110', '1001', '1101'], 5: ['10111', '01010', '11111', '10110', '11110', '01011', '00011', '00010', '10101', '10100']},
        16: {2: ['11', '10', '01'], 3: ['111', '101', '110', '100', '011', '010'], 4: ['1111', '1011', '1001', '1110', '1100', '1010', '0111', '1000', '0101', '0110'], 5: ['11110', '11111', '10111', '10011', '11100', '11000', '10101', '10100', '11101', '01110']},
        17: {2: ['11', '10', '01'], 3: ['110', '111', '101', '100', '011', '010'], 4: ['1100', '1101', '1111', '1001', '1110', '1010', '0111', '1011', '0101', '0110'], 5: ['11000', '11011', '11001', '11111', '11010', '10010', '01111', '10011', '11101', '10100']},
        18: {2: [], 3: ['011', '001'], 4: ['0111', '0011', '0101', '0001', '0110'], 5: ['00111', '01111', '01010', '00110', '01110', '01011', '00010', '01100', '00011']},
        19: {2: [], 3: [], 4: ['0011', '0111', '0101', '0001'], 5: ['00111', '01111', '01110', '01011', '00110', '00011', '00010', '00101']},
        20: {2: [], 3: ['001', '011'], 4: ['0011', '0111', '0010', '0110', '0101', '0001'], 5: ['01111', '00111', '01001', '00001', '00100', '01011', '01101', '00101', '00010', '00110']},
        21: {2: ['11', '10'], 3: ['111', '101', '110', '100', '011', '001'], 4: ['1111', '1100', '1011', '1110', '1010', '1001', '1101', '0011', '0111', '0110'], 5: ['11001', '11111', '00110', '10111', '11100', '10110', '01100', '11101', '01001', '10101']},
        22: {2: [], 3: [], 4: ['0011', '0010'], 5: ['00111', '00100', '00101', '01100', '00110']},
        23: {2: ['11', '10'], 3: ['111', '101', '110', '100', '001'], 4: ['1111', '1101', '1011', '1001', '1110', '1100', '1000', '0011', '0001'], 5: ['11011', '11111', '11001', '11101', '10111', '10100', '10011']},
        24: {2: ['10', '11'], 3: ['101', '110', '100'], 4: ['1100', '1011', '1000', '1010', '1111', '1101', '1001', '1110', '0011'], 5: ['01110', '11000', '10001', '11001', '10111', '10000', '10100', '11100', '10101', '10010']},
        25: {2: ['11', '10'], 3: ['101', '111', '110', '100', '001', '010'], 4: ['1010', '1100', '1111'], 5: ['10100', '11001', '11111', '10111', '10000', '11011', '00100', '01111', '11110', '01011']},
        26: {2: ['10', '11'], 3: ['101', '001', '100', '111'], 4: ['1011', '0010', '1000', '0011', '1010'], 5: ['11000', '10110', '00100', '00101', '10000', '10001', '00110', '10111', '10100']},
        27: {2: ['11', '10'], 3: ['111', '101', '110', '100', '001'], 4: ['1011', '1111', '1110', '1010', '1001', '1101', '1000', '1100', '0110', '0010'], 5: ['01111', '11000', '10111', '11010', '10010', '11111', '01101', '11101', '00101', '10101']},
        28: {2: ['10', '11'], 3: ['100', '101', '111', '110'], 4: ['1000', '1010'], 5: ['10000', '10100', '11001', '10101']},
        29: {2: ['10', '11'], 3: ['101', '111', '001'], 4: ['1111', '1011', '0011', '0111', '1010', '0010', '0101', '0001', '1110', '1101'], 5: ['11111', '10111', '01110', '11110', '00111', '10110', '00110', '10011', '01111', '10101']},
        30: {2: ['10', '11'], 3: ['111', '101', '011', '001', '100', '110'], 4: ['1011', '1111', '0011', '1110', '0111', '1010', '1000', '0010', '0110', '1001'], 5: ['11110', '10110', '10111']},
        31: {2: ['10', '11', '01'], 3: ['101', '111', '100', '011', '001', '110', '010'], 4: ['1011', '1010', '0001', '1110', '1001', '1111', '0011', '1000'], 5: ['10111', '11100', '10100', '10101', '00011', '00010', '10011', '10110', '00110', '10001']},
        32: {2: ['10', '11', '01'], 3: ['111', '101', '100', '011', '110', '001', '010'], 4: ['1010', '1110', '1111', '1011', '0001', '0110', '1001', '0010', '0111', '1000'], 5: ['10100', '11100', '10101', '11101', '00101', '11111', '00011', '10111', '01100', '10010']},
    }

    """
    two = dict()
    three = dict()
    four = dict()
    five = dict()
    for key in combined_models:
        for key2 in combined_models[key]:
            e_list = combined_models[key][key2]
            for i in e_list:
                if key2 == 2:
                    if i in two.keys():
                        two[i] += 1
                    else:
                        two[i] = 1
                if key2 == 3:
                    if i in three.keys():
                        three[i] += 1
                    else:
                        three[i] = 1
                if key2 == 4:
                    if i in four.keys():
                        four[i] += 1
                    else:
                        four[i] = 1
                if key2 == 5:
                    if i in five.keys():
                        five[i] += 1
                    else:
                        five[i] = 1

    am = AnalyticsManager()
    sorted_list = sorted(two.items(), key=lambda x: x[1], reverse=True)
    am.draw_bar_chart(sorted_list, u'Pogostost emisij pri združitvi dveh HMM', "Emisije", u"Število eksperimnetov", "visualize_combined_2")
    sorted_list = sorted(three.items(), key=lambda x: x[1], reverse=True)
    am.draw_bar_chart(sorted_list, u'Pogostost emisij pri združitvi treh HMM', "Emisije", u"Število eksperimnetov", "visualize_combined_3")
    sorted_list = sorted(four.items(), key=lambda x: x[1], reverse=True)
    am.draw_bar_chart(sorted_list, u'Pogostost emisij pri združitvi štirih HMM', "Emisije", u"Število eksperimnetov", "visualize_combined_4")
    sorted_list = sorted(five.items(), key=lambda x: x[1], reverse=True)
    am.draw_bar_chart(sorted_list, u'Pogostost emisij pri združitvi petih HMM', "Emisije", u"Število eksperimnetov", "visualize_combined_5")
    """
    first_stage_attributes = {
        7: [['TTTA', 'ATTT', 'TTTG', 'CGAT', 'TTTC', 'TTTT', 'GTTT', 'TATT', 'CTTT', 'CCTC', 'TTAT'], [2], 0.90965578648847334],
        6: [['GTAC', 'ACAC', 'ACGT', 'CGAC', 'AACG', 'CTTC', 'ACGA', 'AAAC', 'ACAT', 'CTTT', 'AACT', 'TTAA', 'TTCG', 'TTCA', 'TTCC', 'ATTC', 'TTTC', 'CCCT', 'ATAC', 'CAAC', 'CCTT', 'CCCC', 'ATCG', 'CACA', 'ATCA', 'TACC', 'CATA', 'CATC', 'TCAT', 'TAAC', 'TAAA', 'CGTT', 'TATC', 'CATT', 'CGTA'], [0, 2, 3, 5, 6], 0.68526657832326687],
        5: [['GTAC', 'GTAA', 'AAAT', 'ACAT', 'GTAT', 'ATTT', 'TTGT', 'TTAA', 'ATTA', 'TTAT', 'TTTA', 'ATAA', 'ATAC', 'TTTT', 'ATGT', 'TACA', 'ATAT', 'CATA', 'AATT', 'TAAA', 'TGTA', 'TATA', 'TAAT', 'AATA', 'TATT', 'TATG'], [0, 2, 3, 4, 5, 6], 0.89386555869083784],
        4: [['GTAC', 'CGAA', 'ACGT', 'CGAC', 'GTCC', 'ACCG', 'AACG', 'CTTC', 'ACGA', 'CAGA', 'AACT', 'GGAC', 'TTCG', 'GAAG', 'TTCC', 'GCCC', 'CCCT', 'CAAC', 'CCTT', 'CCCC', 'CCAA', 'TACC', 'TACG', 'CATC', 'CGTT', 'CGTC', 'CGCA', 'TCCA'], [0, 2, 3, 4, 5, 6], 0.85446699250232894],
        3: [['ACTT', 'CTTA', 'GTTT', 'GTAT', 'CTTT', 'TTCT', 'ATTT', 'TTAC', 'TTGT', 'TTAA', 'TTAT', 'TTTA', 'TTTC', 'TTTG', 'TTTT', 'ATAT', 'AATT', 'TAAA', 'TCTT', 'TATA', 'TATC', 'TAAT', 'CATT', 'TATT', 'TGTT'], [0, 2, 3, 4, 5, 6], 0.83198514946083613],
        2: [['CGGC', 'CGGA', 'GTCG', 'CGGG', 'AAGA', 'ACCG', 'AACG', 'ACGA', 'ACGG', 'CGAA', 'GGAC', 'CGAC', 'AGCG', 'CCGG', 'GAAG', 'GACG', 'GCCG', 'GCGC', 'CCGC', 'GCGG', 'CCGA', 'TACG', 'TCGA', 'CGCG', 'CGCA'], [0, 2, 3, 5, 6], 0.78727346586374236],
        1: [['GTAC', 'GTAA', 'TATA', 'CTAT', 'GTTT', 'GTAT', 'CTTT', 'TTCT', 'ATTT', 'TTGT', 'TTAA', 'TTAT', 'CGTT', 'TTTA', 'TTTC', 'TTTG', 'TTTT', 'ATAT', 'TACG', 'TAAC', 'TAAA', 'TGTA', 'TCTT', 'ACTT', 'TATC', 'TATT', 'CGTA', 'TATG', 'TGTT'], [0, 2, 3, 4, 5, 6], 0.82454948739366851],
        0: [['CGGC', 'CGGA', 'GTCG', 'CGGG', 'CGAC', 'ACCG', 'AACG', 'GGGG', 'GGGC', 'ACGG', 'CGGT', 'GGCG', 'GGAC', 'AGCG', 'GCGC', 'GAAG', 'GACG', 'GCCG', 'CCGG', 'CCGC', 'GCGG', 'TGCG', 'ACGA', 'CGCT', 'TCGG', 'CGCG', 'CGCA'], [0, 2, 3, 5, 6], 0.72804846793423039],
        15: [['GTAC', 'ACAC', 'AAAT', 'AACA', 'AACG', 'AAAC', 'ACAT', 'TTAC', 'TTAA', 'TTCA', 'ATAA', 'TTAT', 'TTTA', 'CTTA', 'ATAC', 'CACA', 'TACA', 'TACC', 'ATAT', 'TACG', 'CATA', 'TAAC', 'TAAA', 'CGTT', 'TATA', 'TATC', 'CATT', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.69357486284816217],
        14: [['AAAT', 'ACAA', 'TATA', 'AACA', 'CTTA', 'AAAC', 'CTTT', 'AACT', 'TTAC', 'TTAA', 'ATTC', 'ATTA', 'TTAT', 'TTTA', 'ATAA', 'ATAC', 'TACT', 'TACC', 'ATAT', 'TACG', 'TAAC', 'AATT', 'TAAA', 'CGTT', 'ACTT', 'TATC', 'TAAT', 'AATA', 'CATT', 'TATT'], [0, 2, 3, 4, 5, 6], 0.78598558180951028],
        13: [['CGGC', 'CGGA', 'GTCG', 'CGGG', 'CGAC', 'ACCG', 'CGAG', 'CGGT', 'GGCG', 'AGCG', 'CCGG', 'TTCG', 'GACG', 'GCCG', 'GCGC', 'CCGC', 'GCGA', 'GCGG', 'CCGA', 'CCCG', 'TCGC', 'CGCT', 'TCGG', 'CGCG', 'TCCG', 'CGCC', 'CGTC', 'CGCA'], [0, 2, 3, 5, 6], 0.75780017684987355],
        12: [['AAAT', 'ACTT', 'CTTA', 'AAAC', 'GTAT', 'CTTT', 'AACT', 'ATTT', 'TTAA', 'TTAT', 'TTTA', 'TTTC', 'ATAA', 'ATAC', 'TACT', 'ATGT', 'AACG', 'ATAT', 'TAAC', 'AATT', 'TAAA', 'TATA', 'TATC', 'TAAT', 'AATA', 'CATT', 'AATG', 'TATT', 'TCTA', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.84093450049419582],
        11: [['AAAT', 'GTTT', 'CTTT', 'AACT', 'TTCT', 'ATTT', 'TTGT', 'TTAA', 'TTCC', 'TTAT', 'TTTA', 'TTTC', 'CCTT', 'TTTT', 'TACC', 'ATAT', 'TAAC', 'TAAA', 'TGTA', 'TCTT', 'TATC', 'CATT', 'TATT', 'CGTA', 'TCGT', 'TGTT'], [0, 2, 3, 4, 5, 6], 0.9192985203046169],
        10: [['CGGC', 'CGGA', 'GTCG', 'CGAA', 'ACGT', 'CGAC', 'ACCG', 'AACG', 'CTTC', 'ACGA', 'CGGT', 'GGAC', 'AGCG', 'TTCG', 'GCCG', 'CCGC', 'CCGA', 'ATCG', 'TGCG', 'TCGA', 'TACG', 'CGTT', 'CGCT', 'TCGG', 'TCGT', 'CGCG', 'TCCG', 'CGTC', 'CGCA'], [0, 2, 3, 4, 5, 6], 0.84706001534942399],
        9: [['GTAC', 'GTAA', 'CTTA', 'GTTT', 'GTAT', 'CTTT', 'AACT', 'TTCT', 'ATTT', 'TTGT', 'TTAA', 'TTAT', 'TTTA', 'TTTC', 'TTTG', 'GGTA', 'TTTT', 'TACG', 'TAAC', 'TATG', 'TCTT', 'ACTT', 'TATC', 'TATT', 'CGTA', 'TGTA', 'TCGT', 'TGTT'], [0, 2, 3, 4, 5, 6], 0.77131104500421832],
        8: [['CGGC', 'A', 'CGGA', 'GTCG', 'CGAA', 'CGAC', 'ACCG', 'AACG', 'ACGA', 'GGAC', 'GACG', 'GCCG', 'C', 'G', 'CCCT', 'CCGG', 'CCGC', 'GCGG', 'CCGA', 'CCCC', 'T', 'TACG', 'CGCT', 'CGCG'], [0, 2, 3, 4, 5, 6], 0.84818021873665173],
        23: [['GTTA', 'GTAC', 'GTAA', 'AAAT', 'ACTT', 'GTAT', 'CTTA', 'AAAC', 'AAAG', 'AACT', 'ATTT', 'TTAA', 'GAAA', 'GATA', 'TTTA', 'ATAA', 'TACT', 'ATGT', 'ATAT', 'TAAG', 'TAAC', 'AATT', 'TAAA', 'TGTA', 'TATA', 'AATA', 'AATG', 'TATT', 'TATG'], [0, 2, 3, 4, 5, 6], 0.89842749919726339],
        22: [['AAAT', 'ACTT', 'CTTA', 'CTAT', 'GTAT', 'AACT', 'ATTT', 'TTAC', 'TTAA', 'ATTA', 'TTAT', 'TTTA', 'ATAA', 'ATAC', 'TACT', 'ATAT', 'CATA', 'TCAT', 'TAAC', 'AATT', 'TAAA', 'TATA', 'AAAC', 'TATC', 'TAAT', 'AATA', 'CATT', 'TATT'], [0, 2, 3, 4, 5, 6], 0.76655491380353946],
        21: [['CGAT', 'GCGA', 'TTTT', 'CTCG'], [0, 2, 3, 5, 6], 0.92955139949029986],
        20: [['GTTA', 'CTTA', 'GTTT', 'GTAT', 'CTTT', 'TTCT', 'ATTT', 'TTAC', 'TTGT', 'TTAA', 'TTCC', 'TTAT', 'TTTA', 'TTTC', 'TTTT', 'ATGT', 'ATAT', 'TAAC', 'AATT', 'TATG', 'TCTT', 'TATA', 'TATC', 'TAAT', 'CATT', 'TATT', 'TGTT'], [0, 2, 3, 6], 0.86250306009239752],
        19: [['GTCG', 'ACAC', 'CGAC', 'GTCC', 'AACG', 'CTTC', 'ACGA', 'CTTT', 'AACT', 'TTCT', 'TTCC', 'CGTT', 'TTTC', 'CCCT', 'CCTT', 'CCCC', 'ATCG', 'CACA', 'TACC', 'CATC', 'CGCA', 'TCCT', 'CTCT', 'TATC', 'TCTA', 'TCCA', 'TCCC', 'TCGT', 'CGTA'], [0, 2, 3, 5, 6], 0.69881756866752642],
        18: [['GTAC', 'CGAA', 'ACGT', 'CGAC', 'ACCG', 'AACG', 'CTTC', 'ACGA', 'AAAC', 'CTTT', 'AACT', 'GGAC', 'AAGA', 'TTAA', 'GAAG', 'TTCC', 'CCCT', 'CAAC', 'AGAA', 'CCTT', 'CCCC', 'TACC', 'TACG', 'TAAC', 'TAAA', 'CGTT', 'ACTT', 'AATG', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.777318193825],
        17: [['ACAA', 'CTAA', 'AACC', 'AACA', 'CTTA', 'ACAT', 'AACT', 'TTAC', 'TTAA', 'ATTA', 'TTAT', 'TTTA', 'ATAA', 'ATAC', 'TACT', 'ACTA', 'TACC', 'CATA', 'TAAC', 'TAAA', 'ACTT', 'AATC', 'TAAT', 'AATA', 'CATT', 'TATT', 'TCTA'], [0, 2, 3, 4, 5, 6], 0.92536908804721318],
        16: [['CTTA', 'GTTT', 'GTAT', 'CTTT', 'TTCT', 'ATTT', 'TTGT', 'TTAA', 'TTCC', 'TTAT', 'TTTA', 'TTTC', 'TTTG', 'TTTT', 'ATGT', 'ATAT', 'TAAC', 'AATT', 'CGTT', 'TCTT', 'TATC', 'TAAT', 'CATT', 'TATT', 'TATG', 'TGTT'], [0, 2, 3, 4, 5, 6], 0.84386102766491133],
        31: [['CGGC', 'CGGA', 'GTCG', 'CGGG', 'CGAC', 'ACCG', 'AACG', 'ACGA', 'ACGG', 'GGAC', 'AGCG', 'GCGC', 'GACG', 'GCCG', 'CCGG', 'CCGC', 'GCGG', 'CCGA', 'CCCC', 'TGCG', 'TACG', 'CGCT', 'CGCG', 'TCCG', 'CGTC', 'CGCA'], [0, 2, 3, 4, 5, 6], 0.71862780440235163],
        30: [['ACGT', 'GTTG', 'GTGA', 'GTGC', 'GTTT', 'GTAT', 'TGCG', 'GTGT', 'GAGT', 'GAAT', 'TTGT', 'ATGA', 'TGCA', 'TGAA', 'ATGT', 'GCAT', 'GCGT', 'CTGT', 'CATG', 'TGTC', 'TATG', 'TGTG', 'AATG', 'TCTG', 'CGTG', 'TGTA', 'TGTT'], [5], 0.77864842080549512],
        29: [['GTAA', 'AAAT', 'CTAA', 'ACTT', 'AACG', 'AAAC', 'GTAT', 'AACT', 'GGTA', 'TTAA', 'GAAA', 'ATTC', 'CTTA', 'TTAT', 'ACGT', 'TTTA', 'ATAA', 'ATAC', 'TACT', 'TGAA', 'TACG', 'TAAC', 'AATT', 'TAAA', 'CGTT', 'TATA', 'TATC', 'TAAT', 'AATG', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.75242699565839577],
        28: [['GTAC', 'CGAG', 'GTCG', 'CGAA', 'ACGT', 'CGAC', 'ACCG', 'AACG', 'ACGA', 'GTAA', 'GTAT', 'CGGT', 'TGTA', 'AGCG', 'TTCG', 'GAAG', 'GACC', 'TACG', 'GACG', 'AATG', 'GCCG', 'ACGG', 'CCGA', 'ATCG', 'ATGT', 'AAGA', 'CGCT', 'GATG', 'TATG', 'TCGA', 'TCGT', 'CGTC', 'CGCA'], [0, 2, 3, 4, 5, 6], 0.87833511479581372],
        27: [['GTAC', 'ACAC', 'GTCC', 'CTCT', 'ACCC', 'ACTA', 'CTTT', 'TTCA', 'TTCC', 'CACT', 'TTTC', 'CCCT', 'CCTT', 'CCCC', 'CACA', 'CTTC', 'CACC', 'ATCA', 'CCTC', 'CCAC', 'CATC', 'ACTT', 'CGTC', 'TCAC', 'CGCA', 'TCCG', 'ACTC', 'TCCC', 'TCGT', 'TCCA'], [0, 2, 3, 4, 5, 6], 0.83668354412121149],
        26: [['GTTA', 'GTAC', 'CGAG', 'GTCG', 'ACGT', 'CGAC', 'ACCG', 'AACG', 'ACGA', 'GTAA', 'GTAT', 'AAGA', 'TTCG', 'GAAG', 'GATG', 'GACG', 'AATG', 'CGCT', 'CCGA', 'ATCG', 'ATGT', 'TACG', 'TAAA', 'TGTA', 'TCGA', 'CGCA', 'TATG', 'TCGT', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.89913849893163478],
        25: [['CGGC', 'CGGA', 'GTCG', 'CGGG', 'CGAC', 'ACCG', 'AACG', 'GGGG', 'GGGC', 'ACGG', 'CGGT', 'GGAC', 'AGCG', 'GCGC', 'GAAG', 'GACG', 'GCCG', 'CCGG', 'GCGG', 'CCGC', 'AGGG', 'TGCG', 'CGCT', 'CGCG', 'CGCA'], [0, 2, 3, 5, 6], 0.68411911046626395],
        24: [['AAAT', 'CTTA', 'CTTC', 'CTAT', 'GTTT', 'CTTT', 'TTCT', 'ATTT', 'TTAC', 'TTGT', 'TTAA', 'ATTC', 'TTCC', 'TTAT', 'TTTA', 'TTTC', 'TTTG', 'CCTT', 'TTTT', 'TACC', 'ATAT', 'TCAT', 'TAAC', 'AATT', 'TAAA', 'TCTT', 'TATA', 'TATC', 'CATT', 'TATT', 'TCTA', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.919934537732965]
    }

    """
    first_stage_attributes = {
        7: [['A', 'C', 'T', 'G'], [2, 4, 5, 6], 0.99817755192142377, 0.95509121509538275],
        6: [['CGGC', 'AAGG', 'TCGA', 'ACAC', 'ACGG', 'CGAC', 'AACG', 'CTTC', 'CGAT', 'ACAT', 'GTAT', 'CGGG', 'ACGT', 'GGAA', 'GGTA', 'TTCG', 'CGTA', 'TTCA', 'TTCC', 'GCGT', 'GATC', 'GCCG', 'TTTC', 'GCGG', 'CCCT', 'CAAG', 'CAAC', 'ATAC', 'CCGC', 'CCGA', 'CCCC', 'ATCG', 'CACA', 'ATCC', 'CCAC', 'ATAT', 'TACG', 'CATA', 'CATC', 'TCAT', 'TCGC', 'TATG', 'CGCT', 'TATC', 'CCTT', 'CATT', 'TCGT', 'CCCG', 'TCCA', 'TCTC', 'CGTT', 'CGCA', 'TGCG'], [0, 1, 2, 3], 0.99935041667914204, 0.99999928768181656],
        5: [['TGTA', 'TATA'], [0, 3, 4], 0.96845638064371609, 0.99974538504033394],
        4: [['AACG'], [0, 2, 3, 4, 5, 6], 0.88099602420837897, 0.0],
        3: [['ACTT', 'CTTA', 'GTTT', 'GTAT', 'CTTT', 'TTCT', 'ATTT', 'TTGT', 'TTAA', 'TTAT', 'TTTA', 'TTTC', 'TTTG', 'ATAT', 'AATT', 'TAAA', 'TGTA', 'TCTT', 'TATA', 'TATC', 'TAAT', 'CATT', 'TATT', 'TGTT'], [0, 6], 0.93855108977879764, 0.99938278243690892],
        2: [['CGCG'], [0, 2, 3, 5, 6], 0.88183469141015247, 0.0],
        1: [['GTAC', 'GTAA', 'GGTA', 'TTTT', 'TACG', 'TATT', 'TTAT', 'CGTA'], [0, 3, 4, 6], 0.99903378169219859, 0.94479524565482598],
        0: [['GCGG', 'CGAC'], [3], 0.98098019321840302, 0.96843564129970805],
        15: [['CGGA', 'GTCG', 'ACAC', 'ACGT', 'CGAC', 'CTTA', 'ACCC', 'ACGA', 'ACGC', 'ACAT', 'TTAA', 'ACCT', 'CGAA', 'GGAC', 'AGCG', 'TTCC', 'CCCT', 'GCGC', 'AACG', 'ATAC', 'GCGG', 'CCTT', 'CCCC', 'CCCA', 'CACA', 'CACC', 'TACA', 'CCAC', 'CAAT', 'ATAT', 'TACG', 'CATA', 'CATC', 'TCAT', 'TAAC', 'TAAA', 'TCGC', 'CGTT', 'TATA', 'TCAA', 'CATT', 'CGCG', 'CGCA', 'CGTA'], [0, 3, 4, 5, 6], 0.99730921832846597, 0.99999680665765101],
        14: [['TTAA'], [0, 2, 3, 4, 5, 6], 0.89302655248630491, 0.0],
        13: [['CGGC', 'TCGA', 'GCGC', 'CGTT', 'CGCT', 'AACG', 'ACGA', 'TTCG', 'CGCG', 'TACG'], [0, 2, 3, 4], 0.99936665032804095, 0.99999670298835752],
        12: [['TTAA'], [0, 2, 3, 4, 5, 6], 0.86195782201696447, 0.0],
        11: [['TCGT'], [0, 2, 3, 4, 5, 6], 0.888850640450074, 0.99980580774571037],
        10: [['GTAC', 'CGGA', 'GTCG', 'CGAA', 'ACGT', 'ACCG', 'AACG', 'ACGA', 'CGAT', 'ACGG', 'CTTT', 'GGAC', 'AGCG', 'TTCG', 'GACG', 'CCTT', 'CCCC', 'ATCG', 'TACC', 'TACG', 'TCGC', 'CGTT', 'CGCT', 'ACTT', 'TCGG', 'CGTC', 'CGCA', 'CGCG', 'TCCG', 'TATG', 'TCGT', 'CGTA'], [4], 0.58819588205629381, 0.99921787031453224],
        9: [['TTCT', 'GTAA', 'TTTC', 'TTTT', 'GTAT', 'CTTT', 'TATT', 'TGTT', 'GGTA', 'TACG'], [0, 2, 3, 4, 6], 0.99844370555838258, 0.90203758269553169],
        8: [['CGAG', 'GTCG', 'CGAA', 'CGAC', 'ACCG', 'AACG', 'CGGA', 'ACGA', 'CGGT', 'CGGG', 'GGCG', 'AGCG', 'TTCG', 'GACG', 'GCCG', 'GCGC', 'CCGC', 'GCGA', 'GCGG', 'CCGA', 'ATCG', 'TGCG', 'CGTG', 'CCGT', 'CGCT', 'TCGT', 'CGCG', 'TCCG', 'CGTC', 'CGCA', 'CGTA'], [0, 4], 0.94151371601381917, 0.99267357915024701],
        23: [['TTAA', 'TAAA'], [0, 2, 3, 4, 5, 6], 0.89010496862953126, 0.0],
        22: [['TTAA'], [0, 2, 3, 4, 5, 6], 0.91257992535849719, 0.0],
        21: [['A', 'C', 'T', 'G'], [0, 2, 3, 5, 6], 0.99369583741621237, 0.99980998537050481],
        20: [['TAAC'], [0, 2, 3, 5, 6], 0.99126855812846693, 0.94022966068003799],
        19: [['CCGA', 'GTAC', 'CGGA', 'GTCG', 'ACAC', 'CTAC', 'CGAC', 'GTCC', 'ACCG', 'AACG', 'CGAG', 'CTTC', 'ACGA', 'CGAT', 'ACGC', 'AAGT', 'CTTT', 'TTCT', 'CTCG', 'AGCG', 'TTCG', 'TTCC', 'TTTC', 'TGCT', 'CCCT', 'GCGG', 'CCTT', 'CCCC', 'ATCG', 'CACG', 'CACA', 'TGCG', 'CGTG', 'GCAT', 'CCGT', 'TACG', 'CATC', 'TCGC', 'CGTT', 'TCGA', 'CTCT', 'TCGG', 'CGAA', 'ACCC', 'TCGT', 'TCCG', 'TCCA', 'TCTC', 'TCCT', 'CGCA', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.99921089129123863, 0.99999988586041422],
        18: [['ACGT', 'GAAC', 'GTCG', 'TAAC', 'CGAC', 'ATCG', 'TGAA', 'ACCG', 'CCGC', 'CCGA', 'ACGA', 'CGAT', 'TTCG', 'GAAG', 'CGCA', 'CTTC', 'TGCG', 'TACG', 'CGTC', 'TCGT', 'CGTA'], [0, 3, 5, 6], 0.85852504940001628, 0.99931247590844063],
        17: [['CGCG', 'ACTA', 'GTCG', 'TAAC'], [0, 3, 4, 5, 6], 0.96268505885479649, 0.99996664374423805],
        16: [['A', 'C', 'T', 'G'], [4], 0.97811707548746563, 0.97229633271191518],
        31: [['CGGC', 'CGGA', 'GTCG', 'CGGG', 'ACCG', 'ACGA', 'CGGT', 'CGAA', 'GGCG', 'AGCG', 'CCGG', 'TTCG', 'GACG', 'GCCG', 'GCGC', 'GCGG', 'GCGA', 'CCGC', 'CCGA', 'ATCG', 'TGCG', 'GGGG', 'CGCT', 'TCGG', 'TCGT', 'CGCG', 'TCCG', 'CGTC', 'CGCA', 'CGTA'], [0, 2, 3, 4, 5], 0.8978223761236549, 0.99668712414188743],
        30: [['GTGT'], [0, 2, 5, 6], 0.99921769156344353, 0.99698871212451479],
        29: [['AAAC', 'TTAA', 'CGTA'], [0, 2, 3, 4, 5, 6], 0.8727060237398172, 0.0],
        28: [['CGAC'], [0, 2, 3, 4, 5, 6], 0.88004351775199152, 0.0],
        27: [['CTAG', 'CGGA', 'GTAA', 'ACGT', 'AAGA', 'GTGA', 'AGTG', 'GTGG', 'ACGG', 'CTAC', 'AGAG', 'AGAT', 'GATG', 'GACG', 'TGAT', 'CACT', 'ATGA', 'CCGA', 'CCCC', 'ATGT', 'GGCT', 'TACG', 'TGGC', 'TAGT', 'TGGT', 'CGTG', 'ACTG'], [0, 1, 2, 3, 4, 5, 6], 0.99999999276696683, 0.90415613763097602],
        26: [['ACCG', 'ACGA', 'CGAT', 'GACG', 'CGCA', 'TACG'], [0, 3, 4, 5, 6], 0.9355727187917952, 0.93557271879179527],
        25: [['GCGG', 'AGCG', 'CGAC'], [0, 3, 5], 0.974, 0.989],
        24: [['TTTT', 'TCGT', 'TACG'], [0, 2, 3, 4, 5, 6], 0.90704532822835948, 0.99962401244922017]
    }
    """
    """
    t_max = 1
    t_key = 0
    for key in first_stage_attributes.keys():
        dict_l = first_stage_attributes[key]
        if dict_l[3] > 0.0 and dict_l[3] < t_max:
            t_max = dict_l[3]
            t_key = key

    print t_key
    """
    #a = [0.851, 0.858, 0.889, 0.905, 0.871, 0.875, 0.901, 0.890, 0.832, 0.890, 0.845, 0.875, 0.892, 0.876, 0.854, 0.923, 0.923, 0.999, 0.997, 0.999, 0.879, 0.999, 0.838, 0.960, 0.890, 0.860, 0.856, 0.986, 0.916, 0.868, 0.950, 0.868]

    #print sum(a)/float(len(a))

    """
    sum1 = 0
    count1 = 0
    sum2 = 0
    count2 = 0

    for key in first_stage_attributes.keys():
        dict_l = first_stage_attributes[key]
        sum1 += dict_l[2]
        count1 += 1
        if dict_l[3] > 0.0:
            sum2 += dict_l[3]
            count2 += 1

    print "1", sum1 / count1
    print "2", sum2 / count2
    """
    """
    attributes = []
    for key in first_stage_attributes.keys():
        data = first_stage_attributes[key]
        attributes.append(data[2])

    print sum(attributes) / len(attributes)
    """

    """
    am = AnalyticsManager()

    attributes = dict()
    for key in first_stage_attributes.keys():
        data = first_stage_attributes[key]
        for attr in data[0]:
            if attr not in attributes.keys():
                attributes[attr] = 1
            else:
                attributes[attr] += 1

    sorted_list = sorted(attributes.items(), key=lambda x: x[1], reverse=True)

    print len(sorted_list)
    print len([x for x in sorted_list if x[1] > 4])
    print [x for x in sorted_list if x[1] > 1]
    print sorted_list
    #am.draw_bar_chart(sorted_list[0:50], 'Pojavitev kmerov v HMM eksperimentov', "Kmeri", u"Število eksperimnetov", "visualize_kmers")

    a = [('TACG', 11), ('CGTA', 9), ('CGAC', 8), ('ACGA', 8), ('TCGT', 8), ('CGCA', 8), ('TTAA', 7), ('TTCG', 7), ('AACG', 7), ('GTCG', 7), ('GCGG', 7), ('CGCG', 7), ('CGGA', 6), ('CCGA', 6), ('ATCG', 6), ('ACCG', 6), ('AGCG', 6), ('CGAA', 5), ('CGAT', 5), ('ACGT', 5), ('GACG', 5), ('CGTT', 5), ('CCCC', 5), ('TGCG', 5), ('CGCT', 5), ('TAAC', 4), ('CCGC', 4), ('TCGC', 4), ('TTTC', 4), ('CGTC', 4), ('CTTT', 4), ('GCGC', 4), ('CCTT', 4), ('TCCG', 4), ('GTAC', 3), ('GTAA', 3), ('GTAT', 3), ('T', 3), ('ATAT', 3), ('TAAA', 3), ('CGGC', 3), ('CGGG', 3), ('ACGG', 3), ('TTCT', 3), ('TTCC', 3), ('GCCG', 3), ('C', 3), ('G', 3), ('CGTG', 3), ('TCGA', 3), ('TCGG', 3), ('ACAC', 3), ('TTTT', 3), ('TATA', 3), ('TATT', 3), ('CTTC', 3), ('GGTA', 3), ('A', 3), ('CCCT', 3), ('CACA', 3), ('CATC', 3), ('CATT', 3), ('CGAG', 2), ('TTAT', 2), ('ATAC', 2), ('CCAC', 2), ('TGTA', 2), ('ACTT', 2), ('TCTC', 2), ('TATG', 2), ('TGTT', 2), ('ACGC', 2), ('CGGT', 2), ('GGCG', 2), ('CCGT', 2), ('ACAT', 2), ('CTAC', 2), ('GGAC', 2), ('ACCC', 2), ('TCAT', 2), ('TATC', 2), ('CTTA', 2), ('GCGA', 2), ('CATA', 2), ('TCCA', 2), ('AGTG', 1), ('AAAC', 1), ('GAAC', 1), ('GAAG', 1), ('TCTT', 1), ('TAAT', 1), ('ACTA', 1), ('ACTG', 1), ('CTAG', 1), ('AAGA', 1), ('CTCT', 1), ('GTGG', 1), ('AAGT', 1), ('CACG', 1), ('TTCA', 1), ('GATG', 1), ('GATC', 1), ('CCGG', 1), ('ATCC', 1), ('GGCT', 1), ('TGGC', 1), ('TGGT', 1), ('CCCG', 1), ('CTCG', 1), ('AAGG', 1), ('ATGT', 1), ('GGAA', 1), ('ATTT', 1), ('AGAG', 1), ('GTGA', 1), ('TGAT', 1), ('TTTA', 1), ('TTTG', 1), ('CAAG', 1), ('AGAT', 1), ('CAAC', 1), ('GCAT', 1), ('CAAT', 1), ('TAGT', 1), ('AATT', 1), ('TCAA', 1), ('TGAA', 1), ('GTCC', 1), ('GGGG', 1), ('GTTT', 1), ('ACCT', 1), ('GTGT', 1), ('TTGT', 1), ('CACT', 1), ('ATGA', 1), ('TGCT', 1), ('CCCA', 1), ('CACC', 1), ('TACA', 1), ('TACC', 1), ('GCGT', 1), ('TCCT', 1)]
    b = [('TTAA', 17), ('TTTA', 16), ('TTAT', 15), ('TAAC', 15), ('TAAA', 15), ('AACG', 15), ('TATC', 14), ('TATT', 14), ('CTTT', 14), ('CGAC', 13), ('GTAT', 13), ('TACG', 13), ('ATAT', 12), ('ATTT', 12), ('TTTC', 12), ('CTTA', 12), ('AACT', 12), ('GTAC', 11), ('ACTT', 11), ('ACGA', 11), ('TATA', 11), ('CGTA', 11), ('ACCG', 11), ('CATT', 11), ('CGCA', 11), ('CGTT', 10), ('TTTT', 10), ('AATT', 10), ('GTCG', 10), ('AAAT', 9), ('TAAT', 9), ('TTCC', 9), ('TATG', 9), ('GTTT', 9), ('TTGT', 9), ('TACC', 9), ('AAAC', 8), ('ATAA', 8), ('ATAC', 8), ('TGTA', 8), ('ACGT', 8), ('TTCT', 8), ('GACG', 8), ('GCCG', 8), ('GGAC', 8), ('ATGT', 8), ('CGCT', 8), ('GTAA', 7), ('TTAC', 7), ('GAAG', 7), ('TCTT', 7), ('TGTT', 7), ('CGGC', 7), ('CGGA', 7), ('CCGC', 7), ('CCGA', 7), ('CCTT', 7), ('TCGT', 7), ('AATG', 7), ('CTTC', 7), ('AGCG', 7), ('CCCC', 7), ('CGCG', 7), ('CGAA', 6), ('TTCG', 6), ('CCGG', 6), ('TACT', 6), ('TTTG', 6), ('AATA', 6), ('CGTC', 6), ('CCCT', 6), ('GCGG', 6), ('CGGG', 5), ('ACGG', 5), ('CGGT', 5), ('ATCG', 5), ('GCGC', 5), ('TGCG', 5), ('CATA', 5), ('TCTA', 4), ('AAGA', 4), ('TCGA', 4), ('ACAC', 4), ('ACAT', 4), ('ATTC', 4), ('ATTA', 4), ('CACA', 4), ('CATC', 4), ('TCCG', 4), ('CGAG', 3), ('CTAT', 3), ('TTCA', 3), ('TCGG', 3), ('AACA', 3), ('CAAC', 3), ('TCAT', 3), ('GTTA', 3), ('GTCC', 3), ('TCCA', 3), ('CGAT', 2), ('GAAA', 2), ('ACTA', 2), ('CTAA', 2), ('CTCT', 2), ('GGCG', 2), ('GATG', 2), ('ATCA', 2), ('CCTC', 2), ('ACAA', 2), ('TGAA', 2), ('GGGG', 2), ('GGGC', 2), ('GGTA', 2), ('GCGA', 2), ('TACA', 2), ('TCCC', 2), ('GAAT', 1), ('T', 1), ('CCAA', 1), ('CCAC', 1), ('TAAG', 1), ('TGTC', 1), ('TCTG', 1), ('ACTC', 1), ('GTGC', 1), ('AGGG', 1), ('AAAG', 1), ('CGTG', 1), ('TCAC', 1), ('GACC', 1), ('GATA', 1), ('C', 1), ('GCCC', 1), ('G', 1), ('AGAA', 1), ('CAGA', 1), ('TCGC', 1), ('CCCG', 1), ('CTCG', 1), ('CACC', 1), ('GTGA', 1), ('GCGT', 1), ('ACCC', 1), ('GCAT', 1), ('CTGT', 1), ('TGTG', 1), ('AATC', 1), ('GTTG', 1), ('AACC', 1), ('GAGT', 1), ('GTGT', 1), ('A', 1), ('CACT', 1), ('ATGA', 1), ('TGCA', 1), ('CATG', 1), ('TCCT', 1), ('CGCC', 1)]

    c = set([x[0] for x in a])
    d = set([x[0] for x in b])
    u = set.intersection(c, d)
    print u
    print len(u)

    """

    am = AnalyticsManager()

    attributes = dict()
    for key in first_stage_attributes.keys():
        data = first_stage_attributes[key]
        for attr in data[1]:
            if attr not in attributes.keys():
                attributes[attr] = 1
            else:
                attributes[attr] += 1

    sorted_list = sorted(attributes.items(), key=lambda x: x[1], reverse=True)

    key_bindings = {0: 'exon', 1: 'intron', 2: '5UTR', 3: 'ORF', 4: '3UTR', 5: 'ncRNA', 6: 'RNAFold'}

    sorted_list2 = []

    for i in sorted_list:
        sorted_list2.append((key_bindings[i[0]], i[1]))

    print sorted_list2
    am.draw_bar_chart(sorted_list2, 'Pojavitev attributov v HMM eksperimentov', "Atributi", u"Število eksperimnetov", "visualize_attr")


    iclip_experiments = {
        0: ['CLIP-seq-eIF4AIII_2.processed.bedGraph.gz', 'B'],
        1: ['ICLIP_TIA1_hg19.processed.bedGraph.gz', 'P'],
        2: ['CLIPSEQ_SFRS1_hg19.processed.bedGraph.gz', 'M'],
        3: ['PARCLIP_ELAVL1_hg19.processed.bedGraph.gz', 'C'],
        4: ['PARCLIP_IGF2BP123_hg19.processed.bedGraph.gz', 'F'],
        5: ['PARCLIP_PUM2_hg19.processed.bedGraph.gz', 'K'],
        6: ['ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome.processed.bedGraph.gz', 'H'],
        7: ['ICLIP_hnRNPC_Hela_iCLIP_all_clusters.processed.bedGraph.gz', 'G'],
        8: ['PARCLIP_AGO2MNASE_hg19.processed.bedGraph.gz', 'A'],
        9: ['ICLIP_TIAL1_hg19.processed.bedGraph.gz', 'P'],
        10: ['PARCLIP_AGO1234_hg19.processed.bedGraph.gz', 'A'],
        11: ['CLIPSEQ_ELAVL1_hg19.processed.bedGraph.gz', 'C'],
        12: ['PARCLIP_FUS_mut_hg19.processed.bedGraph.gz', 'E'],
        13: ['ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome.processed.bedGraph.gz', 'J'],
        14: ['PARCLIP_EWSR1_hg19.processed.bedGraph.gz', 'D'],
        15: ['ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome.processed.bedGraph.gz', 'H'],
        16: ['ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters.processed.bedGraph.gz', 'Q'],
        17: ['PARCLIP_QKI_hg19.processed.bedGraph.gz', 'L'],
        18: ['CLIPSEQ_AGO2_hg19.processed.bedGraph.gz', 'A'],
        19: ['ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome.processed.bedGraph.gz', 'H'],
        20: ['ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters.processed.bedGraph.gz', 'Q'],
        21: ['ICLIP_HNRNPC_hg19.processed.bedGraph.gz', 'G'],
        22: ['PARCLIP_FUS_hg19.processed.bedGraph.gz', 'E'],
        23: ['PARCLIP_MOV10_Sievers_hg19.processed.bedGraph.gz', 'I'],
        24: ['PARCLIP_ELAVL1A_hg19.processed.bedGraph.gz', 'C'],
        25: ['CLIP-seq-eIF4AIII_1.processed.bedGraph.gz', 'B'],
        26: ['HITSCLIP_Ago2_binding_clusters.processed.bedGraph.gz', 'A'],
        27: ['PARCLIP_RBPMS_hg19.processed.bedGraph.gz', 'R'],
        28: ['HITSCLIP_Ago2_binding_clusters_2.processed.bedGraph.gz', 'A'],
        29: ['PARCLIP_TAF15_hg19.processed.bedGraph.gz', 'N'],
        30: ['ICLIP_TDP43_hg19.processed.bedGraph.gz', 'O'],
        31: ['PARCLIP_ELAVL1MNASE_hg19.processed.bedGraph.gz', 'C']}

    #output_file = gzip.GzipFile('iclip_experiment_groups.pkl', 'wb')
    #output_file.write(pickle.dumps(iclip_experiments, -1))
    #output_file.close()

    chromosome_lengths = {"1": 249250621,
                          "2": 243199373,
                          "3": 198022430,
                          "4": 191154276,
                          "5": 180915260,
                          "6": 171115067,
                          "7": 159138663,
                          "8": 146364022,
                          "9": 141213431,
                          "10": 135534747,
                          "11": 135006516,
                          "12": 133851895,
                          "13": 115169878,
                          "14": 107349540,
                          "15": 102531392,
                          "16": 90354753,
                          "17": 81195210,
                          "18": 78077248,
                          "19": 59128983,
                          "20": 63025520,
                          "21": 48129895,
                          "22": 51304566,
                          "X": 155270560,
                          "Y": 59373566}

    groups = {
        "A": [1, 2, 3, 4, 5],
        "B": [6, 7],
        "C": [8, 9, 10, 11],
        "D": [12],
        "E": [13, 14],
        "F": [15],
        "G": [16, 17],
        "H": [18, 19, 20],
        "I": [21],
        "J": [22],
        "S": [23],
        "K": [24],
        "L": [25],
        "M": [26],
        "N": [27],
        "O": [28],
        "P": [29, 30],
        "R": [31, 32]
    }

    hmm_comparison = {
        1: [(16, 0.90), (4, 0.86), (10, 0.86), (15, 0.86), (31, 0.86), (21, 0.85)],
        2: [(16, 0.88), (4, 0.87), (3, 0.87), (15, 0.85)],
        3: [(4, 0.91)],
        4: [(21, 0.91), (15, 0.91), (1, 0.90), (10, 0.89), (9, 0.89), (3, 0.88), (23, 0.88)],
        5: [(16, 0.89), (17, 0.89), (14, 0.88), (15, 0.88), (31, 0.85), (21, 0.85), (1, 0.84), (9, 0.84), (26, 0.84), (32, 0.83), (20, 0.82), (18, 0.81), (13, 0.81), (22, 0.79), (2, 0.78), (10, 0.78)],
        6: [(32, 0.80), (15, 0.74), (7, 0.68)],
        7: [(15, 0.83), (32, 0.83), (1, 0.79), (11, 0.77), (31, 0.76), (21, 0.75), (16, 0.75), (3, 0.74), (4, 0.74)],
        8: [(16, 0.90), (15, 0.88), (1, 0.88), (31, 0.86)],
        9: [(10, 0.93)],
        10: [(9, 0.94)],
        11: [(32, 0.85), (16, 0.84), (8, 0.84), (15, 0.81), (30, 0.80), (2, 0.80), (26, 0.79), (31, 0.79), (13, 0.73), (7, 0.73), (12, 0.73)],
        12: [(16, 0.90), (15, 0.89), (1, 0.87), (24, 0.87), (31, 0.87), (21, 0.86), (8, 0.84), (32, 0.84), (9, 0.83), (10, 0.83), (14, 0.82), (4, 0.81), (3, 0.81), (23, 0.81), (17, 0.80), (11, 0.79)],
        13: [(16, 0.91), (31, 0.86), (15, 0.85), (32, 0.85), (24, 0.83), (1, 0.83), (8, 0.80), (21, 0.79), (12, 0.79), (14, 0.77)],
        14: [(16, 0.90), (21, 0.90), (10, 0.89), (9, 0.89), (1, 0.89), (23, 0.89), (15, 0.89), (29, 0.88), (3, 0.87), (4, 0.87), (31, 0.86), (24, 0.86), (8, 0.86)],
        15: [(16, 0.89), (21, 0.87),  (4, 0.87)],
        16: [],
        17: [],
        18: [(16, 0.91), (21, 0.89), (15, 0.88), (31, 0.87), (1, 0.87), (23, 0.86), (4, 0.86), (8, 0.86), (3, 0.86), (10, 0.86), (9, 0.86), (24, 0.86), (32, 0.85), (29, 0.83), (17, 0.83), (5, 0.82), (12, 0.81), (14, 0.81), (28, 0.81), (30, 0.80), (26, 0.79), (13, 0.79), (27, 0.77), (11, 0.75), (25, 0.74), (19, 0.73), (20, 0.72), (7, 0.71), (6, 0.69)],
        19: [(16, 0.90), (31, 0.87), (15, 0.86), (1, 0.85), (32, 0.84), (8, 0.84), (21, 0.81), (29, 0.80), (30, 0.79), (23, 0.78), (12, 0.77), (14, 0.76), (28, 0.76), (11, 0.75), (26, 0.75), (13, 0.75), (4, 0.74), (10, 0.74), (2, 0.73), (9, 0.73), (3, 0.73), (24, 0.72), (7, 0.72), (17, 0.71), (27, 0.70)],
        20: [(16, 0.91), (9, 0.90), (10, 0.89), (21, 0.88), (1, 0.87), (31, 0.86), (15, 0.86), (17, 0.86), (24, 0.84), (23, 0.84), (14, 0.83), (8, 0.83), (32, 0.82), (26, 0.82), (4, 0.81), (3, 0.81), (28, 0.80), (12, 0.78), (13, 0.77), (29, 0.75), (30, 0.74), (7, 0.73), (5, 0.73), (22, 0.72), (27, 0.71), (6, 0.71), (25, 0.71)],
        21: [(16, 0.91), (4, 0.90), (9, 0.90), (3, 0.90)],
        22: [(16, 0.92), (21, 0.89), (9, 0.89), (10, 0.88), (15, 0.87), (1, 0.86), (31, 0.86), (12, 0.85), (23, 0.85), (3, 0.84), (8, 0.84), (14, 0.84), (4, 0.84), (29, 0.84), (17, 0.83), (2, 0.82), (30, 0.82), (32, 0.81), (13, 0.80), (27, 0.80), (28, 0.79), (26, 0.78), (25, 0.77), (24, 0.77)],
        23: [(16, 0.90), (10, 0.90), (9, 0.90)],
        24: [],
        25: [(15, 0.96), (1, 0.94), (2, 0.94), (21, 0.92), (16, 0.91), (14, 0.91), (26, 0.90), (23, 0.89), (8, 0.88), (24, 0.88), (31, 0.87), (9, 0.87), (29, 0.87), (4, 0.86), (3, 0.86), (10, 0.86), (32, 0.85)],
        26: [(16, 0.90), (15, 0.87), (21, 0.87), (31, 0.86), (1, 0.85), (4, 0.85), (3, 0.84), (32, 0.84), (8, 0.83), (29, 0.82), (10, 0.82), (30, 0.82), (12, 0.81), (23, 0.81), (14, 0.81), (2, 0.80)],
        27: [(16, 0.90), (24, 0.89), (31, 0.87), (15, 0.85), (9, 0.85), (10, 0.85), (8, 0.84), (32, 0.84), (1, 0.83), (21, 0.83), (17, 0.83), (14, 0.82), (23, 0.81), (12, 0.79), (3, 0.78), (13, 0.78), (4, 0.78), (26, 0.78), (29, 0.77), (30, 0.76), (28, 0.75)],
        28: [(16, 0.89), (15, 0.85), (31, 0.85), (32, 0.83), (1, 0.82)],
        29: [(16, 0.90), (8, 0.87), (21, 0.86), (31, 0.86), (10, 0.86), (17, 0.86), (9, 0.85), (15, 0.85), (23, 0.85), (32, 0.84), (3, 0.83), (4, 0.83), (12, 0.83), (1, 0.83)],
        30: [(16, 0.91), (15, 0.87), (31, 0.86), (1, 0.85), (32, 0.83), (8, 0.82), (17, 0.81), (29, 0.80), (14, 0.80), (21, 0.80), (12, 0.79), (13, 0.78), (26, 0.77), (9, 0.77)],
        31: [(16, 0.91)],
        32: [(16, 0.91), (31, 0.87), (15, 0.86)]
    }

    """
    e_count = dict()
    for key in hmm_comparison.keys():

        e_count[key] = len(hmm_comparison[key])

    am = AnalyticsManager()
    sorted_list = sorted(e_count.items(), key=lambda x: x[1], reverse=True)
    am.draw_bar_chart(sorted_list, u'Število HMM, ki bolje napovedo posamezni eksperiment\n', "Eksperimenti", u"Število eksperimnetov", "visualize_rank3")

    e_count = dict()
    for key in hmm_comparison.keys():

        g = ""
        for j in groups:
            if int(key) in groups[j]:
                g = j

        list_of_e = hmm_comparison[key]
        final_list = []
        for e in list_of_e:
            if e[0] not in groups[g]:
                final_list.append(e)

        e_count[key] = len(final_list)

    am = AnalyticsManager()
    sorted_list = sorted(e_count.items(), key=lambda x: x[1], reverse=True)
    am.draw_bar_chart(sorted_list, u'Število HMM, ki bolje napovedo posamezni eksperiment\n', "Eksperimenti", u"Število eksperimnetov", "visualize_rank4")
    """
    """
    e_count = dict()
    p_list = []
    for key in hmm_comparison.keys():
        list_of_e = hmm_comparison[key]

        for i in list_of_e:
            p_list.append(i[1])
            if i[0] in e_count.keys():
                e_count[i[0]] += 1
            else:
                e_count[i[0]] = 1

    print e_count
    print "avg P", sum(p_list) / len(p_list)
    am = AnalyticsManager()
    sorted_list = sorted(e_count.items(), key=lambda x: x[1], reverse=True)

    print len(sorted_list)
    am.draw_bar_chart(sorted_list, u'Število eksperimentov, ki jih posamezni HMM napove bolje od pripadajočih HMM\n', "Eksperimenti", u"Število eksperimnetov", "visualize_rank1")
    """
    """
    e_count = dict()
    p_list = []
    for key in hmm_comparison.keys():
        list_of_e = hmm_comparison[key]
        g = ""
        for j in groups:
            if int(key) in groups[j]:
                g = j

        for i in list_of_e:
            p_list.append(i[1])
            if i[0] not in groups[g]:
                if i[0] in e_count.keys():
                    e_count[i[0]] += 1
                else:
                    e_count[i[0]] = 1

    print e_count
    print "avg P", sum(p_list) / len(p_list)
    am = AnalyticsManager()
    sorted_list = sorted(e_count.items(), key=lambda x: x[1], reverse=True)

    print len(sorted_list)
    am.draw_bar_chart(sorted_list, u'Število eksperimentov, ki jih posamezni HMM napove bolje od pripadajočih HMM\n', "Eksperimenti", u"Število eksperimnetov", "visualize_rank2")
    """
    # Number of gene sequences
    #seq = pickle.load(gzip.open("E:\Dokumenti\FRI\Magistrska\Results\Generic sequences\make_generic_sequences_from_new_data_0_1.pkl"))
    #print len(seq)

    # Length of first chromosome string
    #string_file = open("E:\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\data\core\chr1.string")
    #dna_string = string_file.read()
    #print len(dna_string)

    # Length of gene sequences combined
    #seq = pickle.load(gzip.open("C:\Aleks\Generic sequences\make_generic_sequences_from_new_data_0_1.pkl"))
    #len_sum = 0
    #for s in seq:
    #    len_sum += len(s[1])
    #print len_sum

    """
    print "all chr"
    clip = {}
    for key in iclip_experiments.keys():
        clip[iclip_experiments[key][0]] = key

    clip_counts = {}
    clip_counts_first = {}
    for i in xrange(32):
        clip_counts[i] = 0
        clip_counts_first[i] = 0

    file_path = "E:\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\data\core\clip_bedGraphs\\"

    clip_counts_all = {}
    clip_counts_all_first = {}
    for file_name in os.listdir(file_path):

        opened_file = gzip.open(file_path + file_name)
        key = clip[file_name]
        index = 0
        for line in opened_file:
            if index > 0:
                if line.split()[0] == "chr1":
                    clip_counts_first[key] += (int(line.split()[2]) - int(line.split()[1]))
                clip_counts[key] += (int(line.split()[2]) - int(line.split()[1]))
            index += 1
        #print file_name, float(clip_counts[key]) / float(sum(chromosome_lengths.values())), clip_counts[key]
        #clip_counts_all[key] = float(clip_counts[key]) / float(sum(chromosome_lengths.values()))
        #clip_counts_all_first[key] = float(clip_counts_first[key]) / float(chromosome_lengths["1"])
    #print clip_counts_all.keys()

    for i in clip_counts.keys():
        print iclip_experiments[int(i)], clip_counts_first[i], clip_counts[i], (float(clip_counts_first[i]) / float(clip_counts[i])) * 100
    """

    """
    file_path = "D:\Dokumenti\FRI\Magistrska\Results\Second stage HMMs Final\\"

    for file_name in os.listdir(file_path):
        if file_name.startswith("data_for_table_cross_validation_of_second_stage_hmm_"):
            opened_file = pickle.load(gzip.open(file_path + file_name))
            print opened_file
    """
    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Make sequences from the start  ##########################################################
    ############################################################################################

    """
    file_manager = FileManager()
    file_manager.gene_file = "..\data\core\hg19_ensembl69_geneunits.tab"
    file_manager.region_file = "..\data\core\hg19_ensembl69_segmentation_segments.tab"
    file_manager.string_file = "..\data\core\chr1.string"

    file_manager.make_data()
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Check iclip distribution per experiment  ################################################
    ############################################################################################

    """
    file_manager = FileManager()
    file_manager.iclip_files_path = "..\data\core\\iclip\\"
    file_manager.iclip_file_name = "ICLIP_chrome-chr1_strand*.pkl"
    list_of_experiments = range(0, 16)

    analytics_manager = AnalyticsManager()
    analytics_manager.distribution_of_iclip_intensities(file_manager, list_of_experiments)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Make generic sequences  #################################################################
    ############################################################################################

    """
    file_manager = FileManager()
    file_manager.region_files_path = "..\data\core\\region\\"
    file_manager.iclip_files_path = "..\data\core\\iclip\\"
    file_manager.rnafold_files_path = "..\data\core\\rnafold\\"
    file_manager.sequence_files_path = "..\data\core\\sequence\\"
    file_manager.iclip_file_name = "ICLIP_chrome-chr1_strand*.pkl"
    file_manager.region_file_name = "REGION_chrome-chr1_strand*.pkl"
    file_manager.rnafold_file_name = "RNAFOLD_chrome-chr1_strand*.pkl"
    file_manager.sequence_file_name = "SEQUENCE_chrome-chr1_strand*.pkl"

    #state_dict = {"1": [0.00001, 1.0], "2": [1.00001, 100000]}
    state_dict = {"1": [0.00001, 100000]}
    order = 1
    #list_of_experiments = [0, 1, 2, 3, 4, 5, 6, 7]
    #list_of_experiments = [8, 9, 10, 11, 12, 13, 14, 15]
    #list_of_experiments = [16, 17, 18, 19, 20, 21, 22, 23]
    #list_of_experiments = [24, 25, 26, 27, 28, 29, 30, 31]
    #list_of_experiments = [0, 1, 2, 3]
    #list_of_experiments = [4, 5, 6, 7]
    #list_of_experiments = [8, 9, 10, 11]
    #list_of_experiments = [12, 13, 14, 15]
    #list_of_experiments = [16, 17, 18, 19]
    #list_of_experiments = [20, 21, 22, 23]
    #list_of_experiments = [24, 25, 26, 27]
    list_of_experiments = [28, 29, 30, 31]
    #list_of_experiments = [7]

    file_manager.make_generic_sequences(state_dict, order, list_of_experiments)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Make generic tri-state sequences  #######################################################
    ############################################################################################

    """
    analytics_manager = AnalyticsManager()

    file_manager = FileManager()
    file_manager.region_files_path = "..\data\core\\region\\"
    file_manager.iclip_files_path = "..\data\core\\iclip\\"
    file_manager.rnafold_files_path = "..\data\core\\rnafold\\"
    file_manager.sequence_files_path = "..\data\core\\sequence\\"
    file_manager.iclip_file_name = "ICLIP_chrome-chr1_strand*.pkl"
    file_manager.region_file_name = "REGION_chrome-chr1_strand*.pkl"
    file_manager.rnafold_file_name = "RNAFOLD_chrome-chr1_strand*.pkl"
    file_manager.sequence_file_name = "SEQUENCE_chrome-chr1_strand*.pkl"

    state_dict1 = {"1": [0.00001, 4.999999999999], "2": [5.0000, 100000]}
    state_dict2 = {"1": [0.00001, 100000]}
    order = 1
    #list_of_experiments = [0, 1, 2, 3, 4, 5, 6, 7]
    #list_of_experiments = [8, 9, 10, 11, 12, 13, 14, 15]
    #list_of_experiments = [16, 17, 18, 19, 20, 21, 22, 23]
    #list_of_experiments = [24, 25, 26, 27, 28, 29, 30, 31]

    #list_of_experiments = [0, 1, 2, 3]
    #list_of_experiments = [4, 5, 6, 7]
    #list_of_experiments = [8, 9, 10, 11]
    list_of_experiments = [12, 13, 14, 15]
    #list_of_experiments = [16, 17, 18, 19]
    #list_of_experiments = [20, 21, 22, 23]
    #list_of_experiments = [24, 25, 26, 27]
    #list_of_experiments = [28, 29, 30, 31]

    #list_of_experiments = [7]

    for e in list_of_experiments:
        distribution = analytics_manager.distribution_of_iclip_intensities(file_manager, [e])
        print distribution
        five_and_over = [x for x in distribution[e].keys() if x >= 5.0]
        zero_and_five = [x for x in distribution[e].keys() if 0.0 < x < 5.0]
        print five_and_over
        if len(five_and_over) > 0 and len(zero_and_five) > 0:
            print "0 1 2"
            file_manager.make_generic_sequences(state_dict1, order, [e])
        else:
            print "0 1"
            file_manager.make_generic_sequences(state_dict2, order, [e])
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Visualize sequence lengths  #############################################################
    ############################################################################################

    """
    sequences = pickle.load(gzip.open("D:\Dokumenti\FRI\Magistrska\Results\Generic sequences\make_generic_sequences_from_new_data_0_1.pkl"))
    analytics_manager = AnalyticsManager()
    analytics_manager.length_of_sequences(sequences)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Visualize single sequence  ##############################################################
    ############################################################################################

    """
    file_manager = FileManager()
    analytics_manager = AnalyticsManager()
    file_manager.annot_files_path = "E:\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\data\core\\region\\"
    file_manager.iclip_files_path = "E:\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\data\core\iclip\\"
    file_manager.rnafold_files_path = "E:\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\data\core\\rnafold\\"


    final_data = pickle.load(gzip.open("E:\Dokumenti\FRI\Magistrska\Results\First stage HMMs Final\\final_generated_sequences_21_iter-3.pkl"))
    final_hmm = pickle.load(gzip.open("E:\Dokumenti\FRI\Magistrska\Results\First stage HMMs Final\\hmm_21_iter-3_final.pkl"))

    hmm = final_hmm[0]
    sequences = final_data[0]
    experiment = 0

    for i in range(len(sequences)):
        #print sequences[i][2].tolist()["iclip_file"]
        analytics_manager.visualize_new_sequence(sequences[i], file_manager, hmm, experiment, ["1"])
    """
    """
    for j in xrange(32):

        final_data = pickle.load(open("E:\Dokumenti\FRI\Magistrska\Results2\First stage hmms\\final_generated_sequences_" + str(j) + ".pkl"))
        final_hmm = pickle.load(open("E:\Dokumenti\FRI\Magistrska\Results2\First stage hmms\\hmm_" + str(j) + "_final.pkl"))

        hmm = final_hmm[0]
        sequences = final_data[0]
        experiment = j

        for i in range(len(sequences)):
            if sequences[i][2].tolist()["iclip_file"] == 'CLIP_chrome-chr1_strand-+_gene_00015_start-1370240_end_-1385068.pkl.gz':
                print "FOUND"
                analytics_manager.visualize_new_sequence(sequences[i], file_manager, hmm, experiment, ["1"])
    """
    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Histogram of nonzero sequences  #########################################################
    ############################################################################################

    """
    generic_sequences_folder = "D:\Dokumenti\FRI\Magistrska\Results\Generic sequences\\"
    filename_starts_with = "make_generic_sequences_from_new_data_"
    state_min_ratio = {"1": 0.0001}
    dump = True
    analytics_manager = AnalyticsManager()
    sequence_max_length = 400000

    analytics_manager.visualize_reduced_sequences(generic_sequences_folder, filename_starts_with, state_min_ratio, dump, sequence_max_length)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Histogram of nonzero tristate sequences  ################################################
    ############################################################################################

    """
    generic_sequences_folder = "E:\Dokumenti\FRI\Magistrska\Results\Tristate generic sequences\\"
    filename_starts_with = "make_generic_sequences_from_new_data_"
    state_min_ratio = {"1": 0.0001, "2": 0.0001}
    dump = True
    analytics_manager = AnalyticsManager()
    sequence_max_length = 400000

    analytics_manager.visualize_reduced_sequences(generic_sequences_folder, filename_starts_with, state_min_ratio, dump, sequence_max_length)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Find best sequences  ####################################################################
    ############################################################################################

    """
    generic_sequences_folder = "E:\Dokumenti\FRI\Magistrska\Results2\Generic sequences\\"
    filename_starts_with = "make_generic_sequences_from_new_data_"
    state_min_ratio = {"1": 0.0}
    dump = True
    analytics_manager = AnalyticsManager()
    #experiment_ids = [0, 1, 2]
    #experiment_ids = [0, 16, 1, 13, 14, 18, 15, 9, 17, 10, 19, 12]
    experiment_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    analytics_manager.find_best_sequences(generic_sequences_folder, filename_starts_with, state_min_ratio, dump, experiment_ids)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Split sequences between train and test  #################################################
    ############################################################################################

    """
    percent_of_train_sequences = 80
    path_to_single_generic_sequence = "E:\Dokumenti\FRI\Magistrska\Data2\make_generic_sequences_from_new_data_0_CLIP-seq-eIF4AIII_2.processed.bedGraph.gz_1.pkl"
    dump = True
    analytics_manager = AnalyticsManager()

    split_sequences = analytics_manager.sequences_split_ids(percent_of_train_sequences, path_to_single_generic_sequence, dump)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  First stage HMM building  ###############################################################
    ############################################################################################

    """
    names0 = [("make_generic_sequences_from_new_data_31_1.pkl", "31", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_30_1.pkl", "30", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_29_1.pkl", "29", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_28_1.pkl", "28", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_27_1.pkl", "27", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_26_1.pkl", "26", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_25_1.pkl", "25", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_24_1.pkl", "24", {"1": 0.0001})]

    names1 = [("make_generic_sequences_from_new_data_23_1.pkl", "23", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_22_1.pkl", "22", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_21_1.pkl", "21", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_20_1.pkl", "20", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_19_1.pkl", "19", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_18_1.pkl", "18", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_17_1.pkl", "17", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_16_1.pkl", "16", {"1": 0.0001})]

    names2 = [("make_generic_sequences_from_new_data_15_1.pkl", "15", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_14_1.pkl", "14", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_13_1.pkl", "13", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_12_1.pkl", "12", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_11_1.pkl", "11", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_10_1.pkl", "10", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_9_1.pkl", "9", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_8_1.pkl", "8", {"1": 0.0001})]

    names3 = [("make_generic_sequences_from_new_data_7_1.pkl", "7", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_6_1.pkl", "6", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_5_1.pkl", "5", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_4_1.pkl", "4", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_3_1.pkl", "3", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_2_1.pkl", "2", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_1_1.pkl", "1", {"1": 0.0001}),
              ("make_generic_sequences_from_new_data_0_1.pkl", "0", {"1": 0.0001})]

    analytics_manager = AnalyticsManager()

    #state_min_ratio = {"1": 0.0, "2": 0.0}
    state_min_ratio = {"1": 0.0}

    percent_of_train_sequences = 80
    percent_of_inner_train_sequences = 80
    number_of_repetitions = 1
    number_of_inner_repetitions = 4
    hmm_type = "discrete"
    kmer_size = 4
    best_kmers = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25]
    #state_neighbourhoods = {"0": 0, "1": 30, "2": 30}
    state_neighbourhoods = {"0": 0, "1": 100}
    annotations_size = 7
    roc_ratio = 0.51
    min_in_state_ratio = 0.00001
    kmer_probability_ratio = 1.1
    annotation_probability_ratio = 1.1
    minimal_distance_between_kmer_ratio = 1.5
    minimal_distance_between_annotation_ratio = 1.5
    dump = True
    evaluation = "MAE"
    sequence_max_length = 400000
    g_sequences_path = "E:\Dokumenti\FRI\Magistrska\Results\Generic sequences\\"
    #sequences_split = pickle.load(open("E:\Dokumenti\FRI\Magistrska\Results2\sequences_split.pkl"))

    # BEST: 0, 6, 7, 13, 15, 16, 19, 20, 25, 30

    #for i in names0:
    #for i in names1:
    #for i in names2:
    #for i in names3:
    for i in names1[4:]:
        print i[1]
        sequences = pickle.load(gzip.open(g_sequences_path + i[0]))
        #print len(sequences)
        #sequences = sequences[0:50]
        #sequences = [sequences[p] for p in sequences_split[0]]
        sequences = analytics_manager.reduce_sequences(sequences, i[2], sequence_max_length, dump, i[1])
        #print len(sequences)
        analytics_manager.cross_validation_of_hmm_with_extraction2(sequences, percent_of_train_sequences, percent_of_inner_train_sequences, number_of_repetitions, number_of_inner_repetitions, hmm_type, kmer_size, best_kmers, state_neighbourhoods, annotations_size, roc_ratio, min_in_state_ratio, kmer_probability_ratio, annotation_probability_ratio, minimal_distance_between_kmer_ratio, minimal_distance_between_annotation_ratio, dump, i[1], evaluation)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  First stage tristate HMM building  ######################################################
    ############################################################################################

    """
    names0 = [("make_generic_sequences_from_new_data_31_1.pkl", "31", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_30_1.pkl", "30", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_29_1.pkl", "29", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_28_1.pkl", "28", {"1": 0.0001, "2": 0.0001})]

    names1 = [("make_generic_sequences_from_new_data_27_1.pkl", "27", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_26_1.pkl", "26", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_25_1.pkl", "25", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_24_1.pkl", "24", {"1": 0.0001, "2": 0.0001})]

    names2 = [("make_generic_sequences_from_new_data_23_1.pkl", "23", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_22_1.pkl", "22", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_21_1.pkl", "21", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_20_1.pkl", "20", {"1": 0.0001, "2": 0.0001})]

    names3 = [("make_generic_sequences_from_new_data_19_1.pkl", "19", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_18_1.pkl", "18", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_17_1.pkl", "17", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_16_1.pkl", "16", {"1": 0.0001, "2": 0.0001})]

    names4 = [("make_generic_sequences_from_new_data_15_1.pkl", "15", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_14_1.pkl", "14", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_13_1.pkl", "13", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_12_1.pkl", "12", {"1": 0.0001, "2": 0.0001})]

    names5 = [("make_generic_sequences_from_new_data_11_1.pkl", "11", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_10_1.pkl", "10", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_9_1.pkl", "9", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_8_1.pkl", "8", {"1": 0.0001, "2": 0.0001})]

    names6 = [("make_generic_sequences_from_new_data_7_1.pkl", "7", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_6_1.pkl", "6", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_5_1.pkl", "5", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_4_1.pkl", "4", {"1": 0.0001, "2": 0.0001})]

    names7 = [("make_generic_sequences_from_new_data_3_1.pkl", "3", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_2_1.pkl", "2", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_1_1.pkl", "1", {"1": 0.0001, "2": 0.0001}),
              ("make_generic_sequences_from_new_data_0_1.pkl", "0", {"1": 0.0001, "2": 0.0001})]

    n0 = [("make_generic_sequences_from_new_data_26_1.pkl", "26", {"1": 0.0001, "2": 0.0001}), ("make_generic_sequences_from_new_data_25_1.pkl", "25", {"1": 0.0001, "2": 0.0001})]
    n1 = [("make_generic_sequences_from_new_data_24_1.pkl", "24", {"1": 0.0001, "2": 0.0001}), ("make_generic_sequences_from_new_data_21_1.pkl", "21", {"1": 0.0001, "2": 0.0001})]
    n2 = [("make_generic_sequences_from_new_data_20_1.pkl", "20", {"1": 0.0001, "2": 0.0001}), ("make_generic_sequences_from_new_data_16_1.pkl", "16", {"1": 0.0001, "2": 0.0001})]
    n3 = [("make_generic_sequences_from_new_data_17_1.pkl", "17_test2", {"1": 0.0001, "2": 0.0001})]


    #names5 = [("make_generic_sequences_from_new_data_10_1.pkl", "10_test_3", {"1": 0.001, "2": 0.0001})]

    analytics_manager = AnalyticsManager()
    percent_of_train_sequences = 80
    percent_of_inner_train_sequences = 80
    number_of_repetitions = 1
    number_of_inner_repetitions = 4
    hmm_type = "discrete"
    kmer_size = 4
    best_kmers = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25]
    state_neighbourhoods = {"0": 0, "1": 100, "2": 100}
    annotations_size = 7
    roc_ratio = 0.51
    min_in_state_ratio = 0.00001
    kmer_probability_ratio = 1.1
    annotation_probability_ratio = 1.1
    minimal_distance_between_kmer_ratio = 1.5
    minimal_distance_between_annotation_ratio = 1.5
    dump = True
    evaluation = "MAE"
    sequence_max_length = 400000
    g_sequences_path = "E:\Dokumenti\FRI\Magistrska\Results\Tristate generic sequences\\"
    #sequences_split = pickle.load(open("E:\Dokumenti\FRI\Magistrska\Results2\sequences_split.pkl"))

    # BEST: 0, 6, 7, 13, 15, 16, 19, 20, 25, 30

    for i in [names2[2]]:
    #for i in names1:
    #for i in names2:
    #for i in names3:
    #for i in names5:
        print i[1]
        sequences = pickle.load(gzip.open(g_sequences_path + i[0]))
        sequences = analytics_manager.reduce_sequences(sequences, i[2], sequence_max_length, dump, i[1])
        analytics_manager.cross_validation_of_hmm_with_extraction2(sequences, percent_of_train_sequences, percent_of_inner_train_sequences, number_of_repetitions, number_of_inner_repetitions, hmm_type, kmer_size, best_kmers, state_neighbourhoods, annotations_size, roc_ratio, min_in_state_ratio, kmer_probability_ratio, annotation_probability_ratio, minimal_distance_between_kmer_ratio, minimal_distance_between_annotation_ratio, dump, i[1], evaluation)

    """
    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Rank predictions of single experiment with first stage anotations or predictions  #######
    ############################################################################################

    """
    final_sequences_path = "E:\Dokumenti\FRI\Magistrska\Results\First stage HMMs Final\\"
    final_sequences_startswith = "final_generated_sequences_"
    hmm_path = "E:\Dokumenti\FRI\Magistrska\Results\First stage HMMs Final\\"
    hmm_startswith = "hmm_"
    hmm_endsswith = "_final"
    generic_sequences_path = "E:\Dokumenti\FRI\Magistrska\Results\Generic sequences\\"
    generic_sequences_startswith = "make_generic_sequences_from_new_data_"
    file_manager = FileManager()
    list_of_hmms = file_manager.make_hmm_list(final_sequences_path, final_sequences_startswith, hmm_path, hmm_startswith, hmm_endsswith, generic_sequences_path, generic_sequences_startswith)
    evaluation = "ROC"

    chosen_state = "1"
    dump = True
    iclip_info_file = "iclip_labels.pkl"
    am = AnalyticsManager()
    for chosen_experiment in [12,13]:
        print "Experiment: " + str(chosen_experiment)
        test_sequences = pickle.load(gzip.open(list_of_hmms[chosen_experiment][0]))[1]
        am.rank_experiment_predictions(list_of_hmms, chosen_experiment, test_sequences, chosen_state, dump, iclip_info_file, evaluation)
    """
    """
    chosen_state = "1"
    chosen_experiment = 19
    #n_test = 100
    test_sequences = pickle.load(open(list_of_hmms[chosen_experiment][0]))[1]
    #test_sequences = random.sample(test_sequences, n_test)[0:2]
    dump = True
    iclip_info_file = "iclip_labels.pkl"
    evaluation = "ROC"

    am = AnalyticsManager()

    # Predictions of first stage hmms
    evaluation = "ROC"
    #am.rank_experiment_from_hmms_build_with_predictions(train_sequences, test_sequences, list_of_hmms, chosen_experiment_id, chosen_state, file_name, False, evaluation)

    # First stage anotations
    am.rank_experiment_predictions(list_of_hmms, chosen_experiment, test_sequences, chosen_state, dump, iclip_info_file, evaluation)
    """

    """
    am = AnalyticsManager()
    labels = pickle.load(open("iclip_labels.pkl"))
    for fi, f in enumerate(glob.glob("rank_of_preditions_sorted_list_*.pkl")):
        experiment = f.split("_")[5].split(".")[0]
        data = pickle.load(open(f))
        am.draw_table2(data, int(experiment), labels, "Experiment " + experiment, "rank_of_preditions_table_" + experiment)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Network of experiments which predictions are better then their own  #####################
    ############################################################################################

    #pickle.load(open("D:\Dokumenti\FRI\Magistrska\Results\Rank of predictions\\rank_of_preditions_sorted_list_9.pkl"))
    """
    rank_files_path = "E:\Dokumenti\FRI\Magistrska\Results\Rank of predictions\\"
    rank_file_startswith = "rank_of_preditions_sorted_list_"
    iclip_info_file = "iclip_labels.pkl"
    analytics_manager = AnalyticsManager()
    top_treshold_predictions = 3
    analytics_manager.create_network_of_experiments_by_predictions(rank_files_path, rank_file_startswith, iclip_info_file, top_treshold_predictions)
    """
    #a = [(7, 0.905), (4, 0.865), (20, 0.961), (10, 0.854), (16, 0.832), (3, 0.815), (21, 0.805), (1, 0.798), (12, 0.798), (23, 0.796), (14, 0.790), (22, 0.778), (2, 0.774), (11, 0.771), (9, 0.771), (5, 0.757), (17, 0.753), (24, 0.752), (8, 0.750), (30, 0.747), (28, 0.742), (26, 0.739), (29, 0.726), (0, 0.725), (25, 0.682), (15, 0.680), (31, 0.677), (6, 0.674), (19, 0.670), (13, 0.639), (18, 0.618), (27, 0.564)]
    #print len(a)
    #with open('rank_of_preditions_sorted_list_9.pkl', 'wb') as f:
    #    pickle.dump(a, f)

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Second stage sequence building  #########################################################
    ############################################################################################

    """
    file_manager = FileManager()
    analytics_manager = AnalyticsManager()

    final_sequences_path = "C:\Aleks\First stage HMMs Final\\"
    final_sequences_startswith = "final_generated_sequences_"

    hmm_path = "C:\Aleks\First stage HMMs Final\\"
    hmm_startswith = "hmm_"
    hmm_endsswith = "_iter-3_final"

    generic_sequences_path = "C:\Aleks\Generic sequences\\"
    generic_sequences_startswith = "make_generic_sequences_from_new_data_"

    rank_of_predictions_path = "C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\\"
    rank_of_predictions_startswith = "rank_of_preditions_sorted_list_"

    experiments_groups = pickle.load(gzip.open("C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\iclip_experiment_groups.pkl"))

    list_of_hmms = file_manager.make_hmm_list(final_sequences_path, final_sequences_startswith, hmm_path, hmm_startswith, hmm_endsswith, generic_sequences_path, generic_sequences_startswith, rank_of_predictions_path, rank_of_predictions_startswith)

    #selected_experiments = [0, 1, 2, 3]
    #selected_experiments = range(32)
    selected_experiments = range(22,24)

    dump = True
    state_min_ratio = {"1": 0.0}
    n_best = 5

    for experiment in selected_experiments:

        print "Experiment: ", experiment

        sequences = pickle.load(gzip.open(list_of_hmms[experiment][0]))
        experiment_group = experiments_groups[experiment][1]
        list_of_the_best = pickle.load(open(list_of_hmms[experiment][3]))

        best = []
        index = 0
        while len(best) < n_best - 1:
            if experiments_groups[list_of_the_best[index][0]][1] != experiment_group:
                best.append(list_of_the_best[index][0])
            index += 1

        final_selected_experiments = [experiment] + best

        #print final_selected_experiments

        #for i in xrange(1, n_best):

        #    print "I: ", i
        #    final_selected_experiments = [experiment] + best[0:i]

        print "Make train sequences"
        chosen_sequences = sequences[0]
        train_sequences = file_manager.make_sequences_from_predictions(chosen_sequences, list_of_hmms, final_selected_experiments, experiment, analytics_manager, "make_sequences_from_predictions_train_" + str(experiment) + "_" + str(len(best)), dump)
        #train_sequences = file_manager.make_sequences_from_iclip(chosen_sequences, list_of_hmms, final_selected_experiments, experiment, "make_sequences_from_predictions_train_" + str(experiment) + "_" + str(len(best)), dump)

        print "Make test sequences"
        chosen_sequences = sequences[1]
        test_sequences = file_manager.make_sequences_from_predictions(chosen_sequences, list_of_hmms, final_selected_experiments, experiment, analytics_manager, "make_sequences_from_predictions_test_" + str(experiment) + "_" + str(len(best)), dump)
        #test_sequences = file_manager.make_sequences_from_iclip(chosen_sequences, list_of_hmms, final_selected_experiments, experiment, "make_sequences_from_predictions_test_" + str(experiment) + "_" + str(len(best)), dump)
    """

    ############################################################################################
    ############################################################################################
    ############################################################################################

    ############################################################################################
    #  Second stage HMM building  ##############################################################
    ############################################################################################

    """
    analytics_manager = AnalyticsManager()
    file_manager = FileManager()
    experimetns = range(12,16)
    file_path = "C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\\"
    file_starts_with = "make_sequences_from_predictions_test_"

    state_neighbourhoods = {"0": 0, "1": 100}
    best_emissions = range(2, 11)
    emissions_probability_ratio = 1.1
    min_in_state_ratio = 0.00001
    hmm_type = "discrete"
    dump = True
    evaluation = "ROC"
    iclip_info_file = "C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\\iclip_labels.pkl"
    chosen_experimetns = file_manager.read_chosen_experiemnt("C:\Aleks\Dropbox\Workspace\PyCharmProjects\iCLIPHMMProject\iCLIPHMM\scripts\\second_stage_data.txt")

    for e in experimetns:

        print "Experiment: ", e

        final_result = 0.5
        result_data = [[], []]

        for file_name in os.listdir(file_path):
            if file_name.startswith(file_starts_with):

                numbers = file_name[len(file_starts_with):len(file_name) - 4].split("_")

                if e == int(numbers[0]):

                    print "Number_of_experiments: ", numbers[2]

                    test_sequences = pickle.load(gzip.open(file_path + file_name))
                    train_sequences = pickle.load(gzip.open(file_path + file_name[:len(file_starts_with) - 5] + "train_" + file_name[len(file_starts_with):len(file_name)]))

                    result = analytics_manager.cross_validation_of_second_stage_hmm2(train_sequences, test_sequences, state_neighbourhoods, best_emissions, emissions_probability_ratio, min_in_state_ratio, "cross_validation_of_second_stage_hmm_" + file_name[len(file_starts_with):len(file_name) - 4], hmm_type, dump, evaluation, chosen_experimetns[str(numbers[0])], iclip_info_file, int(numbers[0]))

                    if len(result[1]) > 0:
                        print result[1]["1"]
                        if result[1]["1"] > final_result:
                            final_result = result[1]["1"]
                            result_data = result

        if len(result_data[1]) > 0:
            print "Extracted variables: ", result_data[0]
            print "Best result:", result_data[1]
    """
    ############################################################################################
    ############################################################################################
    ############################################################################################

    elapsed = (time.time() - start)
    print "#"*100 + " END " + "#"*100
    print "Time elapsed: ", datetime.timedelta(seconds=elapsed), " ([H]H:MM:SS[.UUUUUU])"


if __name__ == "__main__":
    main()