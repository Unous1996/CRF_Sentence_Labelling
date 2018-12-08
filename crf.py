import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log, exp
import copy

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Class definition for a bigram HMM
class CRF:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        pass
        ### Initialize the rest of your data structures here ###

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        pass
    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        trellis = [[1.0 for x in range(len(self.tags_list))] for y in range(2)]
        backpointer = [[-1 for x in range(len(words))] for y in range(len(self.tags_list))]
        return_list = []

        for tag in self.tags_list:
            temp_prob = self.__getLogStartObservedProb(tag)
            trellis[0][self.tags_list.index(tag)] = temp_prob

        for i in range(len(words)):
            for j in range(len(self.tags_list)):
                log_max_prob = 1.0
                if not words[i] in self.single_word or self.single_word[words[i]] < self.minFreq:
                    log_temp_prob = self.__getLogObservedProb(word = UNK, tag = self.tags_list[j])
                    if log_temp_prob != 1.0:
                        trellis[1][j] = log_temp_prob
                    else:
                        continue
                else:
                    log_temp_prob = self.__getLogObservedProb(word = words[i], tag = self.tags_list[j])
                    if log_temp_prob != 1.0:
                        trellis[1][j] = log_temp_prob
                    else:
                        continue

                for k in range(len(self.tags_list)):
                    log_temp_prob = self.__getLogTransitionProb(currtag=self.tags_list[j], prevtag=self.tags_list[k])
                    if trellis[0][k] != 1.0 and log_temp_prob != 1.0:
                        if log_max_prob == 1.0:
                            log_max_prob = trellis[0][k] + log_temp_prob
                            #max_index_list[j] = k
                            backpointer[j][i] = k
                        else:
                            if log_temp_prob + trellis[0][k] > log_max_prob:
                                log_max_prob = trellis[0][k] + log_temp_prob
                                #max_index_list[j] = k
                                backpointer[j][i] = k

                trellis[1][j] += log_max_prob

            #max_index = trellis[1].index(max(trellis[1], key=self.__helperfun))
            #return_list.append(self.tags_list[max_index])
            max_index_list = [-1 for x in range((len(self.tags_list)))]
            trellis[0] = copy.deepcopy(trellis[1])
            trellis[1] = [1.0 for i in range(len(self.tags_list))]

        #backpropagation phase

        prev_max_index = trellis[0].index(max(trellis[0], key=self.__helperfun))
        return_list.append(self.tags_list[prev_max_index])
        for i in range(len(words)-1,0,-1):
            return_list.insert(0, self.tags_list[backpointer[prev_max_index][i]])
            prev_max_index = backpointer[prev_max_index][i]

        return return_list# this returns a dummy list of "NULL", equal in length to words

if __name__ == "__main__":
    tagger = CRF()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
