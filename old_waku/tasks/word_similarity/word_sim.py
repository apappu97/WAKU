import numpy as np
import scipy
import pickle
from scipy.stats import *
from sklearn.model_selection import train_test_split

'''
annotated_pairs data set is from followinf paper:
Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2001. Placing search in context: the concept revisited. In Proc. of WWW.
'''

def loadSimilarlityData(sim_pairs_filepath):
	"""loads in annotated pairs into dictionary of 'test' and 'val'"""
	data = {}
	tmp = open(sim_pairs_filepath).readlines()
	data['words'] = [ row.strip().split('\t')[0:2] for i, row in enumerate(tmp) if i!=0 ]
	data['sim_scores'] = [ float(row.strip().split('\t')[2]) for i, row in enumerate(tmp) if i!=0 ]
	sim_scores_test, sim_scores_val, words_test, words_val = train_test_split(data['sim_scores'], data['words'], test_size=0.2, random_state=1)
	output = {'test': {'sim_scores': sim_scores_test, 'words': words_test}, 
	          'val': {'sim_scores': sim_scores_val, 'words': words_val}}
	return output

def get_embedding_word_vector_dict(dict_pickle_path, embedding_npz_path):
	"""loads ditionary of word to embedding vector """
    emb_dict = pickle.load(open(dict_pickle_path, 'rb'))
    emb_weight = np.load(embedding_npz_path)['a']

	word_vec_dict = {}

	for key, value in emb_dict.items():
			word_vec_dict[key] = emb_weight[value]
	
	return word_vec_dict


def getSimilarity(e1, e2):
    """computes cosine similarity (cosine of angle between embedding vectors)"""
    return np.sum(e1 * e2)/( np.sqrt(np.sum(e1*e1)) * np.sqrt(np.sum(e2*e2)))
 
def getSimilarityScoreForWords(word_vec_dict, w1,w2):
    if (w2 not in word_vec_dict) or (w1 not in word_vec_dict):
        return -1
    else:
        finalVector_w1 = word_vec_dict[w1]
        finalVector_w2 = word_vec_dict[w2]
        return getSimilarity(finalVector_w1, finalVector_w2)

##### 3) computing correlation between human-annotated scores and cosine similarities for word embeddings
def evaluate(dict_pickle_path, embedding_npz_path, sim_data, mode, verbose=False):
    """
    mode is either 'test' or 'val'
    """
    word_vec_dict = get_embedding_word_vector_dict(dict_pickle_path, embedding_npz_path)
    data = sim_data[mode]

    pred_scores = []
    invalid = 0

    # loop through human annotated data, returning matrix where col 0 is cosine similarity of embeddings, and col 1 is human score
    pred_scores = [[getSimilarityScoreForWords(word_vec_dict, w1w2[0],w1w2[1]), human_score] for w1w2, human_score in zip(data['words'], data['sim_scores'])]

    # delete word pairs which couldn't be found in embedding set
    pred_scores = np.array( [ val for val in pred_scores if val[0] != -1])

    sp_rank_coeff, sp_rho = spearmanr(pred_scores[:,0], pred_scores[:,1])
    
    if verbose:
        print("total, valid, spearman_rank_coeff, sp_rho", len(data['words']),len(pred_scores), sp_rank_coeff, sp_rho)

    return sp_rank_coeff, sp_rho

# annotated_pairs = '/content/gdrive/My Drive/UCL_ML/NLP Class/WGL/human_sim.txt'
# pairs_data = loadSimilarlityData(annotated_pairs)

# evaluate(dict_pickle_path, embedding_npz_path, pairs_data, 'val', verbose=False)

