from nltk.tag import CRFTagger
import os.path
import sys

def readLabeledData(inputFile):
    if os.path.isfile(inputFile):
        file = open(inputFile, "r")
        sens = [];
        for line in file:
            raw = line.split()
            sentence = []
            for token in raw:
                parts = token.split('_')
                sentence.append((parts[0], parts[1]))
            sens.append(sentence) # append this list as an element to the list of sentences
        return sens
    else:
        print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

def readUnlabelledData(inputFile):
    if os.path.isfile(inputFile):
        file = open(inputFile, "r")  # open the input file in read-only mode
        sens = [];
        for line in file:
            sentence = line.split()  # split the line into a list of words
            sens.append(sentence)  # append this list as an element to the list of sentences
        return sens
    else:
        print(
            "Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit()

if __name__ == "__main__":
    debug = False
    if not debug:
        input_file = sys.argv[1]
    else:
        input_file = "../small_train.txt"

    training_opt = {'c1': 0.0, 'c2': 1.0}
    #training_opt = {'c1':0.0, 'c2': 1.0}
    #traingin_opt = {'c1':1.0', 'c2': 0.0}

    ct = CRFTagger(training_opt = training_opt)
    labelled_data = readLabeledData(inputFile=input_file)
    ct.train(labelled_data, 'model.crf.tagger')

    testfile = "../test.txt"
    test_data = readUnlabelledData(inputFile=testfile)

    goldfile = "../gold.txt"
    gold_data = readLabeledData(inputFile=goldfile)
    ct.tag_sents(test_data)

    print("Result = ", ct.evaluate(gold_data))

