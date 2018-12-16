from nltk.tag import CRFTagger
import os.path
import sys
import re
import unicodedata
import time

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

def get_features(tokens, idx):

    token = tokens[idx]
    
    feature_list = []

    punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])

    if idx == len(tokens) - 1:
        feature_list.append('FIRST_WORD')

    '''
    if idx > 0:
        previous_token = tokens[idx-1]

        feature_list.append('PREV_WORD_' + previous_token)

        if previous_token.isupper():
            feature_list.append('PREVOIOUS_CAPITALIZATION')

            pattern = re.compile(r'\d') 

            if re.search(pattern, token) is not None:
                feature_list.append('PREVOIS_HAS_NUM')

            if all(unicodedata.category(x) in punc_cat for x in previous_token):
                feature_list.append('PREVIOUS_PUNCTUATION')

            if len(previous_token) > 1:
                feature_list.append('PREV_SUF_' + previous_token[-1:])

            if len(previous_token) > 2:
                feature_list.append('PREV_SUF_' + previous_token[-2:])

            if len(previous_token) > 3:
                feature_list.append('PREV_SUF_' + previous_token[-3:])
    
    
    if idx < len(tokens)-1:
        previous_token = tokens[idx+1]
        
        feature_list.append('PREV_WORD_' + previous_token)

        if previous_token.isupper():
            feature_list.append('PREVOIOUS_CAPITALIZATION')

            pattern = re.compile(r'\d') 

            if re.search(pattern, token) is not None:
                feature_list.append('PREVOIS_HAS_NUM')

            if all(unicodedata.category(x) in punc_cat for x in previous_token):
                feature_list.append('PREVIOUS_PUNCTUATION')

            if len(previous_token) > 1:
                feature_list.append('PREV_SUF_' + previous_token[-1:])

            if len(previous_token) > 2:
                feature_list.append('PREV_SUF_' + previous_token[-2:])

            if len(previous_token) > 3:
                feature_list.append('PREV_SUF_' + previous_token[-3:])
    '''

    if not token:
        return feature_list
    
    # Capitalization
    if token[0].isupper():
        feature_list.append('CAPITALIZATION')
    
    # Number
    pattern = re.compile(r'\d') 

    if re.search(pattern, token) is not None:
        feature_list.append('HAS_NUM')
    
    # Punctuation
    if all(unicodedata.category(x) in punc_cat for x in token):
        feature_list.append('PUNCTUATION')
    
    # Suffix up to length 3
    if len(token) > 1:
        #feature_list.append('PREF_' + token[0])
        feature_list.append('SUF_' + token[-1:])
    if len(token) > 2:
        #feature_list.append('PREF_' + token[:2])
        feature_list.append('SUF_' + token[-2:])
    if len(token) > 3:
        #feature_list.append('PREF_' + token[:3])
        feature_list.append('SUF_' + token[-3:])
    
    feature_list.append('WORD_' + token)
    return feature_list

if __name__ == "__main__":
    debug = False
    if not debug:
        input_file = sys.argv[1]
    else:
        input_file = "../small_train.txt"

    ct = CRFTagger(feature_func = get_features)
    labelled_data = readLabeledData(inputFile=input_file)
    start = time.time()
    ct.train(labelled_data, 'model.crf.tagger')
    end = time.time() 

    elapsed = end - start

    print("Total training time = ", elapsed)

    testfile = "../test.txt"
    test_data = readUnlabelledData(inputFile=testfile)

    goldfile = "../gold.txt"
    gold_data = readLabeledData(inputFile=goldfile)
    ct.tag_sents(test_data)

    print("Result = ", ct.evaluate(gold_data))

