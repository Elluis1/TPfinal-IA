import numpy as np

class HammingNetwork:
    def __init__(self, names, patterns):
        self.names = names
        self.patterns = patterns

    def classify(self, input_vector):
        distances = [np.sum(np.abs(p - input_vector)) for p in self.patterns]
        best_index = np.argmin(distances)
        return self.names[best_index], distances[best_index]
