import numpy as np

def inter_dist(W, b, k):
    """
    Calculates average distance between top k words in dimension and intruder.

    Parameters:
    -----------
    W : `numpy.ndarray`
        (k, 300) embeddings of k words
    b : `numpy.ndarray`
        (1, 300) embedding of intruder b
    k : `int`
        Number of words in calculation

    Returns:
    --------
    dist/k : `float`
        Average distance
    """
    dist = 0
    for i in range(k):
        dist += np.linalg.norm(W[i,:] - b)
    return dist/k

def intra_dist(W, k):
    """
    Calculates average distance between top k words in a dimension

    Parameters:
    -----------
    W : `numpy.ndarray`
        (k, 300) embeddings of k words
    k : `int`
        Number of words in calculation
    Returns:
    --------
    dist/(k*(k-1) : `float`
        Average distance
    """    
    dist = 0
    for i in range(k):
        for j in range(k):
            if j != i:
                dist += np.linalg.norm(W[i,:] - W[j,:])
    return dist/(k*(k-1))

def top_ten_set(embedding):
    """
    Gives indices of words that appear in the top 10% of a dimension.

    Parameters:
    -----------
    embedding : `numpy.ndarray`
        The embedding matrix.    
    
    Returns:
    --------
    top_list : `list` of `int`
        List of indices
    """    
    D = embedding.shape[1]
    tenth = embedding.shape[0]//10

    top_list = set()
    # Get words that appear in the 10% of a dimension
    for i in range(D):
        indices = np.argsort(embedding[:,i])[::-1]
        top_list |= set(indices[:tenth])
    return set(top_list)

def dist_ratio(embedding, top_list, k, N, acc_filepath=None, verbose=True):
    """
    Calculates the Intrusion score, namely interDist/intraDist.

    Parameters:
    -----------
    embedding : `numpy.ndarray`
        The embedding matrix.
    top_list : `list` of `int`
        List of indices for words that appear in the top 10% of some dimension.
    k : `int`
        Number of words to compare intruder to.
    N : `int`
        Number of runs to average over
    acc_filepath : `str`
        The output filepath for saving the accuracies.        
    verbose : `Boolean`
        Whether to print.
        
    Returns:
    --------
    results : `dict`
        Dictionary of mean and std of intrusion scores
    """    
    D = embedding.shape[1]
    half = embedding.shape[0]//2

    scores = np.zeros(N)
    
    for run in range(N):
        dist_ratio = 0
        # calculate dist ratio
        for i in range(D):
            indices = np.argsort(embedding[:,i])[::-1]
            topk = indices[:k]
            W = embedding[topk,:]
            
            # pick intruder word
            in_list = False
            while in_list == False:
                bottom_half = indices[half:]
                intruder = np.random.choice(bottom_half)
                in_list = intruder in top_list
            b = embedding[intruder,:]

            # calculate intra + inter dist
            interDist = inter_dist(W, b, k)
            intraDist = intra_dist(W, k)
            
            dist_ratio  += interDist/intraDist
        
        # store dist_ratio
        scores[run] = dist_ratio/D
    
    results = dict()
    results['mean'] = np.mean(scores)
    results['std'] = np.std(scores)

    if verbose:
        print("mean word intrusion: {}, std: {}".format(results['mean'], results['std']))

    if acc_filepath is not None:
        with open('{}/{}.txt'.format(acc_filepath,'word_intrusion_' + str(datetime.now())), 'w') as out:
            out.write("Mean word intrusion: {}, std: {}".format(results['mean'], results['std']))

    return results

def evaluate(embedding, k, N, acc_filepath=None, verbose=False):
    """
    Runs word intrusion experiment.

    Parameters:
    -----------
    embedding : `numpy.ndarray`
        The embedding matrix.
    k : `int`
        Number of words to compare intruder to.
    N : `int`
        Number of runs to average over
    acc_filepath : `str`
        The output filepath for saving the accuracies.        
    verbose : `Boolean`
        Whether to print.
        
    Returns:
    --------
    result['mean'] : `float`
        Mean of N runs of intrusion calculations
    """    
    top_ten = top_ten_set(embedding)
    results = dist_ratio(embedding, 
                         top_ten, k, N, 
                         verbose=verbose, acc_filepath=acc_filepath)
    return results['mean']