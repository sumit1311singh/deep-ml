import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    N = len(corpus)
    result = []

    for doc in corpus:
        scores = []

        for word in query:
            tf = doc.count(word) / len(doc) if len(doc) > 0 else 0

            df = sum(1 for d in corpus if word in d)

            idf = np.log((N + 1) / (df + 1)) + 1

            scores.append(tf * idf)

        result.append(scores)

    return result