import numpy as np

def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    result = []
    predictions = np.array(predictions)
    n=predictions.shape[1]

    for idx in range(n):
        votes=[]
        
        for clf_idx in range(len(predictions)):
            votes.append(predictions[clf_idx][idx])

        labels, counts = np.unique(votes, return_counts=True)
        max_count=np.max(counts)
        winners = labels[counts==max_count]
        result.append(int(np.min(winners)))

    return result
        