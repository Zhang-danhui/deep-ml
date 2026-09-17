import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores=np.array(scores)
    fenzi=np.exp(scores-np.max(scores))
    fenmu=np.sum(fenzi)
    return  fenzi/fenmu