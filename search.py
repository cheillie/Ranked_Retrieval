#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import math

TEST_MODE = False

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    # clear the out_file
    open(results_file, 'w').close()

    stemmer = nltk.stem.porter.PorterStemmer()

    # load dictionary into memory
    dictionary = {}
    with open(dict_file, 'r') as dic_file:
        num_docs = int(dic_file.readline())       # first line in dict_file contains number of docs
        for line in dic_file:
            term, doc_freq, pointer, num_bytes = line.split(" ") 
            dictionary[term] = [int(doc_freq), int(pointer), int(num_bytes)]

    with open(queries_file) as file:
        for line in file:
            # obtain the log tf score of all documents
            tf_wt = {}
            get_tf_wt_doc(stemmer, dictionary, tf_wt, line, postings_file)
            # obtain the normalized score of all documents
            normalized_docs = calc_normalized_docs(tf_wt)

            # obtain the tf-idf score of the query terms
            tf_raw = {}
            tf_wt_dictionary = {}
            idf_dictionary = {}
            tf_idf_dictionary = {}
            calc_tf_idf_query(line, stemmer, dictionary, num_docs, tf_raw, tf_wt_dictionary, idf_dictionary, tf_idf_dictionary)

            #obtain the normalized score of the query terms
            normalized_query = {}
            calc_normalized_query(tf_idf_dictionary, normalized_query)

            # calculate the cosine scores and return the top 10 docIDs
            answer = calc_cosine(normalized_docs, normalized_query)

            # write top 10 answers to results_file
            result_f = open(results_file, "a")
            result_f.write(answer + "\n")
            result_f.close()


def get_tf_wt_doc(stemmer, dictionary, tf_wt_dictionary, line, postings_file):
    """
    obtain the tf_wt_dictionary that contains the tf-wt frequency of each document terms.
    The tf_wt_dictionary will contain the key:value pair where the key is the term, and the value
    is a list of [docID, tf-wt] pair
    e.g. {'hello': [[95190, 0.89]]}
    """
    stemmed_line = []
    for token in nltk.word_tokenize(line):
        stemmed_token = stemmer.stem(token)
        stemmed_line.append(stemmed_token)
    for key in dictionary.keys():
        if key in stemmed_line:
            # obtain the pointer to the posting
            pointer = dictionary[key][1]
            num_bytes = dictionary[key][2]
            # obtain the corresponding posting pointed by the pointer
            with open(postings_file) as postings_f:
                postings_f.seek(pointer, 0)
                posting_list = get_posting(postings_f.read(num_bytes))
            tf_wt_dictionary[key] = [list(x) for x in posting_list]
                
def calc_tf_wt(dictionary_raw):
    """
    calculate the tf-wt score of the query.
    For document terms the dictionary will contain a key:value pair where the key is the term, and 
    ther value is a list of [docID, tf-wt] pair. e.g. {'hello': [[95190, 1.3]]}
    """
    dictionary = {}
    for key, val in dictionary_raw.items():
        dictionary[key] = 1 + math.log(val, 10)
    return dictionary
    
def calc_tf_idf_query(line, stemmer, dictionary, num_docs, tf_raw, tf_wt_dictionary, idf_dictionary, tf_idf_dictionary):
    """
    stem the tokens and calculate the tf-idf score of the query terms in tf_idf_dictionary.
    The tf_idf_dictionary contains the key:value pair where the key is the term, and the value is the tf-idf score
    e.g.  {'hello': 0.69}
    """
    for token in nltk.word_tokenize(line):
        stemmed_token = stemmer.stem(token)
        if stemmed_token in dictionary.keys():
            # obtain the raw tf score of the query term
            tf_raw[stemmed_token] = tf_raw.get(stemmed_token, 0) + 1
            doc_freq = dictionary[stemmed_token][0]
            # obtain the idf score of the query term
            idf = math.log((num_docs/doc_freq) , 10)
            idf_dictionary[stemmed_token] = idf

    # obtain the log tf score of the query term
    tf_wt_dictionary = calc_tf_wt(tf_raw)

    # obtain the tf-idf score of the query terms
    for key in idf_dictionary:
        if key in tf_wt_dictionary:
            tf_idf_dictionary[key] = idf_dictionary[key] * tf_wt_dictionary[key]

def calc_normalized_query(tf_idf_dictionary, normalized_query):
    """
    calculate the normalized score of the query in normalized_query given tf_idf_dictionary
    """
    power = 0
    for val in tf_idf_dictionary.values():
        power += pow(val, 2)
        denom = math.sqrt(power)
    for key, val in tf_idf_dictionary.items():
        normalized = val / denom
        normalized_query[key] = normalized

def calc_normalized_docs(log_tf):
    """
    calculate the normalized score of the document terms, and eturn a normalized_docs dictionary
    where the key is the docID, and the value is a [term, normalized score] pair
    """
    # load the doc length file into memory
    doc_length_dict = {}
    with open("doc_lengths.txt") as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split(",")
            docID = int(splitted[0])
            doc_length = float(splitted[1].replace("\n",""))
            doc_length_dict[docID] = doc_length

    # revise the format of the log_tf dict so that it's easier to calculate normalized score
    lnc_dictionary = {}
    for key, val in log_tf.items():
        for docID_tf in val:
            lnc_dictionary[docID_tf[0]] = lnc_dictionary.get(docID_tf[0], [])
            lnc_dictionary[docID_tf[0]].append([key, docID_tf[1]])
    
    # calculate normalized score
    normalized_docs = {}
    for key, val in lnc_dictionary.items():
        for tf_wt in val:
            wt = tf_wt[1]
            denom = doc_length_dict[key]
            normalized = wt / denom
            normalized_docs[key] = normalized_docs.get(key, [])
            normalized_docs[key].append([tf_wt[0], normalized])

    return normalized_docs

def calc_cosine(normalized_docs, normalized_query):
    """
    calculate the cosine score given the normalized docs dictionary and normalized query dictionary
    """
    scores = {}
    for key, vals in normalized_docs.items():
        score = 0
        for val in vals:
            normalized_query_val = normalized_query[val[0]]
            normalized_doc_val = val[1]
            score += normalized_query_val * normalized_doc_val
        scores[key] = score
    
    if TEST_MODE:
        checklist = [int(x) for x in "1151 8605 11384 8590 11164 5326 10263 11764 11052 11772".split(" ")]
        for i in checklist:
            print(f"doc {i}: {scores[i]}")
            
    scores = sorted(scores, key=scores.get, reverse=True)[:10]

    score_str = ' '.join([str(score) for score in scores])
    return score_str

def get_posting(postings):
    """
    Parse the postings string into a list of tuples. Where the first item in the tuple is the docID, 
    and the second item in the tuple is the term freq
    e.g. "(2,3) (3,10) (4,5)" -> [(2, 3), (3, 10), (4, 5)]
    """
    p = []
    postings = "".join(postings.rstrip())
    postings = postings.split(" ")
    for posting in postings:
        p.append(eval(posting))
    return p


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:t')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    elif o == '-t':
        TEST_MODE = True
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
