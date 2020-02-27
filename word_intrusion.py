import numpy as np

def intra_dist(W, k):
    assert W.shape[0] == k

    dist = 0
    for i in range(k):
        J = [x for x in range(k) if x != i]
        for j in J:
            dist += np.linalg.norm(W[i,:] - W[j,:])
    
    return dist/(k*(k-1))

def inter_dist(W, b, k):
    assert W.shape[0] == k

    dist = 0
    for i in range(k):
        dist += np.linalg.norm(W[i,:] - b)

    return dist/k

def dist_ratio(embedding_weights, k):
    D = embedding_weights.shape[1]
    tenth = embedding_weights.shape[0]//10
    half = embedding_weights.shape[0]//2

    top_list = set()
    # get words that appear in the 10% of a dimension
    for i in range(D):
        indices = np.argsort(embedding_weights[:,i])
        top_list |= set(indices[:tenth])
        if i%50==0:
            print('dim {} complete'.format(i))
    
    dist_ratio = 0
    # calculate dist ratio
    for i in range(D):
        indices = np.argsort(embedding_weights[:,i])
        W = embedding_weights[indices[:5],:]
        
        # pick intruder word
        in_list = False
        while in_list == False:
            intruder = np.random.choice(indices[half:])
            in_list = intruder in top_list
        b = embedding_weights[intruder,:]
        
        # calculate intra + inter dist
        interDist = inter_dist(W, b, k)
        intraDist = intra_dist(W, k)
        dist_ratio += interDist/intraDist

        if i%50==0:
            print('dim {} complete'.format(i))

    return dist_ratio/D