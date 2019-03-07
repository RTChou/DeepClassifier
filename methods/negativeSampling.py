import numpy as np

class NegativeSampling:
    @staticmethod
    def get_label_pairs(labels, unlabeled='unlabeled'):
        """
        labels: training labels before one-hot encoding
        """
        ind = []; lab = []; pos = []; neg = []
        for i in range(len(labels)):
            if labels[i].item() != unlabeled:
                ind.append(i)
                lab.append(labels[i].item())
        for j in range(len(lab)):
            for k in range(len(lab)):
                if lab[j] == lab[k]:
                    pos.append([ind[j], ind[k]])
                else:
                    neg.append([ind[j], ind[k]])
        return (pos, neg)

    @staticmethod
    def sample_context_dist(inst, graph, labels, r1, r2, q, d, pair_sets):
        """
         graph: knn graph with distances as edges
        labels: training labels before one-hot encoding
            r1: ratio of positive and negative samples
            r2: ratio of two types of context
             q: walk length
             d: window size
        """
        i = 0
        c = 0
        gamma = 0
        if np.random.rand(1) < r1:
            gamma = 1
        else:
            gamma = -1
        if np.random.rand(1) < r2:
            s = inst.random_walk(graph, q)
            smpl_pair = inst.sample_nodes(s, d)
            i = smpl_pair[0]
            c = smpl_pair[1]
            if gamma == -1:
                c = np.random.randint(len(graph), size=1).item()
        else:
            if gamma == 1:
                smpl_pair = pair_sets[0][np.random.randint(len(pair_sets[0]), size=1).item()]
                i = smpl_pair[0]
                c = smpl_pair[1]
            else:
                smpl_pair = pair_sets[1][np.random.randint(len(pair_sets[1]), size=1).item()]
                i = smpl_pair[0]
                c = smpl_pair[1]
        return (i, c, gamma)
    
    def random_walk(self, graph, q):
        """
        graph: knn graph with distances as edges
            q: walk length
        """
        s = []
        if q < 1:
            return s
        w = np.random.randint(len(graph), size=1).item()
        s.append(w)
        for i in range(q - 1):
            # convert distance to prob
            sim = 1 / (1 + graph[w])
            p = sim / sum(sim)
            w = np.random.choice(len(graph), 1, p=p).item()
            s.append(w)
        return s
    
    def sample_nodes(self, s, d):
        """
        s: random walk sequence
        d: window size
        """
        node_pairs = []
        for j in range(len(s)):
            for k in range(len(s)):
                if abs(s[j] - s[k]) < d:
                    node_pairs.append([s[j], s[k]])
        ind = np.random.randint(len(node_pairs), size=1).item()
        return node_pairs[ind]

