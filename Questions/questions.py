import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    import os
    corpus = dict()

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding='utf8') as file:
                corpus[filename] = file.read()

    return corpus

    


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    import string
    words = nltk.word_tokenize(document)

    return [
        word.lower() for word in words
        
        if word not in nltk.corpus.stopwords.words("english")
       
        and not all(char in string.punctuation for char in word)
    ]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    import math
    count=dict()
    for sentence in documents:
        already=set()
        for words in documents[sentence]:
            if words not in already:
                already.add(words)
                try:
                    
                    count[words] +=1
                except KeyError:
                    count[words]=1  
    return {words: math.log(len(documents)/count[words]) for words in count}
    

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    tf_idfs = dict()
    returnlist=list()

    for file in files:
        tf_idfs[file] = 0
        for word in query:
            tf_idfs[file] += files[file].count(word) * idfs[word]
    for word,value in sorted(tf_idfs.items(), key=lambda item: item[1], reverse=True)[:n]:
        returnlist.append(word)
        
    return returnlist
    



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    morty = list()
    returnlist=list()

    for sentence in sentences:
        td_idf = [sentence, 0, 0]

        for word in query:
            if word in sentences[sentence]:
                
                td_idf[1] += idfs[word]
                
                td_idf[2] += sentences[sentence].count(word) / len(sentences[sentence])

        morty.append(td_idf)
    
        
    
    for sentence,idk,td in sorted(morty, key=lambda item: (item[1], item[2]), reverse=True)[:n] :
        returnlist.append(sentence)
    return [sentence ]
    
    

if __name__ == "__main__":
    main()
