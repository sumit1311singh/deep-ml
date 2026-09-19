import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        self.classes, counts = np.unique(y, return_counts = True)

        self.priors = {}
        for cls, count in zip(self.classes, counts):
            self.priors[cls] = count / len(y)

        self.feature_probs={}
        for cls in self.classes:        
            mask = y==cls
            x = X[mask]

            cls_count = np.sum(x, axis=0)

            prob = (cls_count + self.smoothing)/(len(x) + 2*self.smoothing)
            self.feature_probs[cls]=prob

    def predict(self, X):
        # Predict class labels for test set X
        predictions = []
        for cls in self.classes:
            prediction = np.log(self.priors[cls]) + np.sum(np.where(
                                                                X==1, 
                                                                np.log(self.feature_probs[cls]), 
                                                                np.log(1-self.feature_probs[cls])
                                                            ),
                                                            axis=1
                                                        )
            
            predictions.append(prediction)

        return np.argmax(predictions, axis=0)        