import numpy as np

def distance2vote(d, a=3, b=5):
    """
    Returns the weight of the vote based on the distance.
    """
    sim = np.maximum(0, 1 - d ** 2 / 2)
    return np.exp(-d ** a) * sim ** b

print(1 - distance2vote(0))