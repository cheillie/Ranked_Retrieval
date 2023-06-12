# Ranked Retrieval

## Commands
Indexing: `python index.py -i directory-of-documents -d dictionary-file -p postings-file`\
Searching: `python search.py -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results`

Documents to be indexed are stored in directory-of-documents. In this homework, we are going to use the Reuters training data set provided by NLTK. .../nltk_data/corpora/reuters/training/

## Notes

### Python Version

We're using Python Version 3.8.10 for this assignment.

### General Notes about this assignment
index.py:
Here are my indexing steps:
1) parse all documents from in_dir
    - these are parsed in increasing order to ensure this property is propagated to the postings lists 
2) construct inverted index from the document tokens 
    - in the indexing phase, we calculate the logarithmic term frequency so that we can later calculate
      document lengths in step 4. Doing both of these steps in indexing will also allow the searching to
      run faster.
3) sort the index
    - this is important so that the line number pointers can later be converted to byte offset pointers
4) calculate the document lengths
    - this is the square root of the sum of squared logarithmic frequencies for each document's
      unique terms. 
5) store the index to disk
    - since we removed HW2's scalable index construction implementation, we now flush the entire index
      from memory
6) convert the line number pointers for postings list addresses to byte offset pointers
    - while indexing, it is easier to work with line numbers/list indices. However, once all indexing 
      is complete, we convert these values into byte addresses within the postings file so that search.py
      can use low-level I/O methods like seek() and read() to optimize searching time.
7) finally, we add the total number of documents to the top of the dictionary  
    - this is to facilitate the idf calculations required for search.py

search.py \
The search algorithms is mainly divided into two parts. The first part is to obtain the normalized score
for the document terms, and the second part is to obtain the normalized score for the query terms.

Calculare normalized score for document terms:
1) obtain the tf-wt score for all the document terms. This value can be obtained from the doc_lengths.txt.
Ignore the terms that are not in the query, since their normalized score will evaluate to 0. 
2) calculate the normalized score given the tf-wt score dictionary 

Calculate normalized score for query terms:
1) calculate the tf-idf score of the query terms:
    a) count the number of times each query term shows up to get the term_freq and calculate tf-wt
    b) obtain the doc_freq from the dictionary
    c) calculate log(num_docs/doc_freq) to get idf.
    d) then multiply tf-wt with idf to get td-idf score
2) calculate the normalized score given the tf-idf dictionary

Once we have both the normalized dictionaries for document terms and query terms, we can
obtain the cosine score and rank them from decreasing order to return the top 10 results.# Ranked_Retrieval
