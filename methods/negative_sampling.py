import numpy as np

def sample_context_dist(graph, labels, r1, r2, q, d):
        i = 0
        c = 0
        gamma = 0
        if np.random.rand(1) < r1:
            gamma = 1
        else:
            gamma = -1
        if np.random.rand(1) < r2:
            s = random_walk(graph, q)
            smpl_pair = sample_nodes(s, d)
            i = smpl_pair[0]
            c = smpl_pair[1]
            if gamma == -1:
                c = np.random.randint(len(graph), size=1).item()
        else:
            pair_sets = get_label_pairs(labels)
            if gamma == 1:
                smpl_pair = pair_sets[0][np.random.randint(len(pair_sets[0]), size=1).item()]
                i = smpl_pair[0]
                c = smpl_pair[1]
            else:
                smpl_pair = pair_sets[1][np.random.randint(len(pair_sets[1]), size=1).item()]
                i = smpl_pair[0]
                c = smpl_pair[1]
        return (i, c, gamma)

def random_walk(graph, q):
        s = []
        if q < 1:
            return s
        w = np.random.randint(len(graph), size=1).item()
        s.append(w)
        for i in range(q - 1):
            p = (1 / (1 + graph[w])) / sum(1 / (1 + graph[w])) # convert distance to prob
            w = np.random.choice(len(graph), 1, p=p).item()
            s.append(w)
        return s

def sample_nodes(s, d):
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

def get_label_pairs(labels):
        ind = []; lab = []; pos = []; neg = []
        for i in range(len(labels)):
            if labels[i] != '':
                ind.append(i)
                lab.append(labels[i].item())
        for j in range(len(lab)):
            for k in range(len(lab)):
                if lab[j] == lab[k]:
                    pos.append([ind[j], ind[k]])
                else:
                    neg.append([ind[j], ind[k]])
        return (pos, neg)

