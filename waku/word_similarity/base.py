import numpy as np
import scipy
from scipy.stats import *

'''
annotated_pairs data set is from followinf paper:
Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2001. Placing search in context: the concept revisited. In Proc. of WWW.
'''

def load_sim_data(sim_data_path, dataset):
    """
    Loads the similarity dataset.
    
    Parameters:
    -----------
    sim_data_path : `str`
        Filepath of the dataset.
    dataset : `str`
        Either `WordSim353` or `SimLex999`.
    
    Returns:
    --------
    data : `dict`
        The dataset with the word pairs and corresponding similarity scores.
    """
    data = dict()
    tmp = open(sim_data_path).readlines()
    data['words'] = [row.strip().split('\t')[0:2] for i, row in enumerate(tmp) if i!=0]
    if dataset == 'SimLex999':
        data['sim_scores'] = [float(row.strip().split('\t')[3]) for i, row in enumerate(tmp) if i!=0]
    elif dataset == 'WordSim353':
        data['sim_scores'] = [float(row.strip().split('\t')[2]) for i, row in enumerate(tmp) if i!=0]
    else:
        raise ValueError('invalid dataset')
    return data
 
def get_sim_score(word2embedding, word1, word2):
    """
    Computes cosine similarity score.
    (cosine of angle between embedding vectors)    
    
    Parameters:
    -----------
    word2embedding : `dict`
        Dictionary mapping the word to its corresponding vector embedding.  
    word1 : `str`
        The first word.
    word2 : `str`
        The second word.
    
    Returns:
    --------
    sim_score : `float`
        The cosine similarity score of the two words.
    """
    if (word2 not in word2embedding) or (word1 not in word2embedding):
        return -1
    else:
        e1 = word2embedding[word1]
        e2 = word2embedding[word2]
        return np.sum(e1*e2) / (np.sqrt(np.sum(e1*e1)) * np.sqrt(np.sum(e2*e2)))

def evaluate(embedding, sim_data, word2index, verbose=False):
    """
    Run word similarity test on the embedding using sim_data.
    
    Parameters:
    -----------
    embedding : `numpy.ndarray`
        The embedding matrix.
    sim_data : `dict`
        The dataset with the word pairs and corresponding similarity scores.
    word2index : `dict`
        Dictionary mapping word to index.       
    verbose : `Boolean`
        Whether to print.
    
    Returns:
    --------
    correlation : `float`
        The spearman rank correlation coefficient for the dataset.
    pvalue : `float`
        The two-sided p-value for a hypothesis test whose null hypothesis 
        is that two sets of data are uncorrelated.
    """        
    word2embedding = {word:embedding[index] for word, index in word2index.items()}
    pred_scores = list()
    invalid = 0
    # Loop through human annotated data, returning matrix where col 0 is cosine similarity 
    # of embeddings, and col 1 is human score
    pred_scores = [[get_sim_score(word2embedding, w1w2[0], w1w2[1]), 
                    human_score] for w1w2, human_score in zip(sim_data['words'], sim_data['sim_scores'])]
    # Delete word pairs which couldn't be found in embedding set
    pred_scores = np.array([val for val in pred_scores if val[0] != -1])
    sp_rank_coeff, sp_rho = spearmanr(pred_scores[:,0], pred_scores[:,1])
    if verbose:
        print("total, valid, spearman_rank_coeff, sp_rho", len(data['words']),len(pred_scores),sp_rank_coeff, sp_rho)
    return sp_rank_coeff, sp_rho