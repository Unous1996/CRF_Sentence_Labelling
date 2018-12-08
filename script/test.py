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
                parts = token.split('_');
                sentence.append((parts[0], parts[1]))
            sens.append(sentence) # append this list as an element to the list of sentences
        return sens
    else:
        print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

if __name__ == "__main__":
    inputfile = sys.argv[1]
    ct = CRFTagger()
    labelled_data = readLabeledData(inputFile=inputfile)
    ct.train(labelled_data, 'model.crf.tagger')
    ct.tag_sents([['dog','is','good'], ['Cat','eat','meat']])
    gold_sentences = [[('dog','Noun'),('is','Verb'),('good','Adj')] , [('Cat','Noun'),('eat','Verb'), ('meat','Noun')]]
    print("Result = ", ct.evaluate(gold_sentences))

