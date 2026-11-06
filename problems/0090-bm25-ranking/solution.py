
import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus or not query:
        raise ValueError("Corpus and query must not be empty")

    N = len(corpus)

    # Calculate document lengths and average length
    doc_lengths = [len(doc) for doc in corpus]
    avgdl = sum(doc_lengths) / N

    # Calculate document frequencies
    df = Counter()
    for doc in corpus:
        df.update(set(doc))

    scores = []

    for doc, dl in zip(corpus, doc_lengths):
        tf = Counter(doc)
        score = 0.0

        for term in query:
            f = tf[term]
            nt = df[term]

            if f == 0:
                continue

            idf = np.log((N + 1) / (nt + 1))

            denominator = f + k1 * (
                1 - b + b * dl / avgdl
            )

            score += idf * (f * (k1 + 1)) / denominator

        scores.append(score)

    return np.round(np.array(scores), 3)
