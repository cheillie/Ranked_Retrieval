#!/usr/bin/python3
import getopt
import linecache
import nltk
import os
import re
import shutil
import sys
import traceback

from collections import Counter, defaultdict, deque
from heapq import merge
from itertools import islice
from math import floor, log, sqrt
from typing import Counter, DefaultDict, Deque, Dict

TEST_MODE = False
VERBOSE = False
TEST_NUM_FILES = 10
AUXILIARY_DICT = 'd'
AUXILIARY_POST = 'p'

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file -t (optional flag for test mode)")


class DictionaryEntry:
    '''
    Represents a term dictionary entry that contains document frequency 
    and the pointer to associated postings list in the postings file
    '''
    def __init__(self, postings_address, doc_freq=1):
        self.doc_freq = doc_freq
        self.postings = postings_address
        
class Index:
    '''
    Inverted index data structure containing a dictionary of terms and the associated postings lists. 
    '''
    def __init__(self):
        ''' 
        Constructor
        '''
        self.term_dictionary: Dict[str, DictionaryEntry] = {}
        self.postings: [Counter] = []
        
    def __len__(self):
        '''
        override built-in length function to help track block size
        '''
        return len(self.term_dictionary)


    def insert(self, term:str, doc_ID:int, doc_freq:int=1, term_freq:int=1):
        if term in self.term_dictionary.keys():
            term_pointer = self.term_dictionary[term].postings
            if doc_ID not in list(self.postings[term_pointer]):
                self.term_dictionary[term].doc_freq += 1
            self.postings[term_pointer].update({doc_ID: term_freq})
        else:
            term_pointer = len(self.postings)
            self.term_dictionary[term] = DictionaryEntry(term_pointer)
            self.postings.append(Counter({doc_ID: term_freq}))
            
    def termwise_sort(self):
        '''
        in-place sort the index by term in alphabetical order
        '''
        pairs = list(zip(self.term_dictionary.items(), self.postings))
        pairs.sort(key=lambda p: (p[0], p[1]))
        sorted_dict, self.postings[:] = zip(*pairs)
        self.term_dictionary = dict(sorted_dict)
        return self
    
    def term_freq_logarithmise(self):
        logarithmised_postings:[Dict] = []
        for postings_list in self.postings:
            logarithmised_postings_list = {}
            for doc_ID, term_freq_raw in postings_list.items():
                logarithmised_postings_list[doc_ID] = log(term_freq_raw, 10) + 1
            logarithmised_postings.append(logarithmised_postings_list)
        self.postings = logarithmised_postings


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    index = construct_index(in_dir)
    index.term_freq_logarithmise()
    index.termwise_sort()
    generate_doc_lengths(index)
    write_index_to_disk(index, out_dict, out_postings)
    convert_to_byte_offset(out_dict, out_postings)
    add_total_count(in_dir, out_dict)

def construct_index(in_dir):
    '''
    Parse all files in in_dir, construct and return inverted term index
    '''
    stemmer = nltk.stem.porter.PorterStemmer()      # one persistent stemmer object
    file_list = os.listdir(in_dir)                  # obtain list of document names
    file_list.sort(key=lambda f: int(f))            # sort document names in algebraically increasing order
    
    index = Index()                                 # initialize the index object
    num_files = 0                                   # counter for running test mode, which parses only 100 files
    if VERBOSE: print(f"Constructing index object...")
    for file in file_list:
        if TEST_MODE and num_files == TEST_NUM_FILES:
            break
        num_files += 1
        with open(f"{in_dir}/{file}", "r") as doc:
            for line in doc:
                for token in tokenize(stemmer, line):                           # tokenize and stem the files
                    if TEST_MODE and VERBOSE: print(f"Inserting {token} into index...")
                    index.insert(term=token, doc_ID=int(file))
    return index           
        
def tokenize(stemmer, line):
    '''
    Given a stemmer object and a list of words,
    return a list of these words tokenized and stemmed using
    nltk.word_tokenize and the porter stemming algorithm
    '''
    tokenized = nltk.word_tokenize(line)
    # UNCOMMENT THIS IF PUNCTUATION REMOVAL IS DESIRED
    # pattern = r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]+'
    # tokenized = filter(lambda x: not re.fullmatch(pattern, x), tokenized)
    return [stemmer.stem(token) for token in tokenized]

def write_index_to_disk(index: Index, out_dict, out_postings):
    '''
    From the given Index object, save the term_dict and postings 
    to disk in 'out_dict' and 'out_postings' respectively.
    '''
    dictionary = index.term_dictionary
    postings = index.postings
    with open(out_dict, "w") as dictionary_file:
        for term, entry in dictionary.items():
            dictionary_file.write(format_dict_entry(term, entry.doc_freq, entry.postings))
    with open(out_postings, "w") as postings_file:
        for posting_list in postings:
            list_to_write = ""
            for doc_ID, term_freq in posting_list.items():
                list_to_write += format_postings_entry(doc_ID, term_freq)
            postings_file.write(f"{list_to_write.strip()}\n")
            
def format_dict_entry(term, doc_freq, postings):
    '''
    return a formmated string representation of a given dictionary entry
    '''
    return f"{term} {doc_freq} {postings}\n"

def format_postings_entry(doc_ID, term_freq):
    '''
    returns a formatted string representation of a given postings entry
    '''
    return f"({doc_ID},{term_freq}) "

def add_total_count(in_dir, out_dict):
    '''
    Add a count of all document IDs to the dictionary file to
    facilitate calculating idf in search.py
    '''
    file_list = os.listdir(in_dir)
    file_count = len(file_list)
    with open(out_dict, "r+") as dictionary_file:
        dictionary_content = dictionary_file.read()
        dictionary_file.seek(0,0)
        dictionary_file.write(f'{file_count}\n{dictionary_content}')
    
def convert_to_byte_offset(out_dict, out_postings):
    '''
    Converts the line-number pointers into byte-offset pointers
    '''
    with open(out_dict, "r") as term_dict, \
        open(out_postings, "r") as postings, \
        open('temp', "w") as out_dict_file:
        total_offset = 0
        for dict_entry, posting_list in zip(term_dict, postings):
            term, doc_freq, line_num = dict_entry.strip().split(" ")
            postings_list_len = len(posting_list.encode('utf-8'))
            out_dict_file.write(f"{term} {doc_freq} {total_offset} {postings_list_len}\n")
            total_offset += postings_list_len
    os.rename(out_dict, 'old_dict')
    os.rename('temp', out_dict)
    os.remove('old_dict')
    
def generate_doc_lengths(index):
    '''
    Calculate the document lengths
    '''
    postings = index.postings
    doc_lengths = defaultdict(float)
    for p_list in postings:
        for doc_ID, term_freq in p_list.items():
            doc_lengths[doc_ID] += term_freq ** 2
    with open('doc_lengths.txt', 'w') as file:
        for doc_ID, length in doc_lengths.items():
            file.write(f"{doc_ID},{sqrt(length)}\n")

input_directory = "/user/e/e1025440/nltk_data/corpora/reuters/training/"
output_file_dictionary = 'dictionary.txt'
output_file_postings = 'postings.txt'

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:tv')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    elif o == '-t': # test mode
        TEST_MODE = True
        print("Running in test mode...")        
    elif o == '-v': # verbose mode
        VERBOSE = True
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)


if __name__ == "__main__":
    build_index(input_directory, output_file_dictionary, output_file_postings)
