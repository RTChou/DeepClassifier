from keras import backend as K
from keras.engine.topology import Layer


def constructGraph():
    """
    k-nearest neighbor approach?
    build a Kernel
    """

def deepWalk(graph, w, d, gamma, t):
    """
    w: window size
    d: embedding size
    gamma: walks per vertex
    t: walk length
    """



def sampleContextDist(graph, labels, r1, r2, q, d):
    """
    r1: ratio of positive and negative samples
    r2: ratio of two types of context
    q: walk length
    """

    # initialization
    i = 1
    c = 1
    gamma = 1
    if np.random.rand(1) < r1:
        gamma = 1
    else:
        gamma = -1
    if np.random.rand(1) < r2:
        S = deepWalk(graph, w, d, gamma, q)
        # uniformly sample (Sj, Sk) with |j - k| < d
        i = Sj
        c = Sk
        if gamma == -1:
            # uniformly sample c from 1:L+U
    else:
        if gamma == 1:
            # uniformly sample (i,c) with yi = yc
        else:
            # uniformly sample (i,c) with yi != yc

    return((i, c, gamma))


class Embedding(Layer):
    
    def __init__(self, output_dim, graph, **kwargs):
        self.output_dim = output_dim
        self.graph = graph
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        this function defines weights
        """



    def call(self, x):
        """
        this function calculates weights
        sampling context distribution will be put here
        """

    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.output_dim)

    
