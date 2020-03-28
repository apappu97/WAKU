import numpy as np
from datetime import datetime
import re
from os import path

def inter_dist(W, b, k):
    assert W.shape[0] == k

    dist = 0
    for i in range(k):
        dist += np.linalg.norm(W[i,:] - b)

    return dist/k

def intra_dist(W, k):
    assert W.shape[0] == k

    dist = 0
    for i in range(k):
        for j in range(k):
            if j != i:
                dist += np.linalg.norm(W[i,:] - W[j,:])
    
    return dist/(k*(k-1))

def top_ten_set(embedding_weights):
    D = embedding_weights.shape[1]
    tenth = embedding_weights.shape[0]//10

    top_list = set()
    # get words that appear in the 10% of a dimension
    for i in range(D):
        indices = np.argsort(embedding_weights[:,i])[::-1]
        top_list |= set(indices[:tenth])

    return set(top_list)

def dist_ratio(embedding_weights, top_list, k, N, print_result=True, save_acc=True, file_path=None):
    D = embedding_weights.shape[1]
    half = embedding_weights.shape[0]//2

    scores = np.zeros(N)
    
    for run in range(N):
        dist_ratio = 0
        # calculate dist ratio
        for i in range(D):
            indices = np.argsort(embedding_weights[:,i])[::-1]
            top5 = indices[:5]
            W = embedding_weights[top5,:]
            
            # pick intruder word
            in_list = False
            while in_list == False:
                bottom_half = indices[half:]
                intruder = np.random.choice(bottom_half)
                in_list = intruder in top_list
            b = embedding_weights[intruder,:]

            # calculate intra + inter dist
            interDist = inter_dist(W, b, k)
            intraDist = intra_dist(W, k)
            
            dist_ratio  += interDist/intraDist
        
        # store dist_ratio
        scores[run] = dist_ratio/D
    
    results = {}
    results['mean'] = np.mean(scores)
    results['std'] = np.std(scores)

    if print_result:
        print("mean word intrusion: {}, std: {}".format(results['mean'], results['std']))

    if save_acc:
        with open('{}/{}.txt'.format(file_path,'word_intrusion_' + str(datetime.now())), 'w') as out:
            out.write("Mean word intrusion: {}, std: {}".format(results['mean'], results['std']))

    return results