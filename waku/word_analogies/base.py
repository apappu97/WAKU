import numpy as np
import io
import time
from collections import defaultdict

import torch

def get_word_id(word, word2id, lower):
    """
    Get a word ID.
    If the model does not use lowercase and the evaluation file is lowercased,
    we might be able to find an associated word.
    """
    assert type(lower) is bool
    word_id = word2id.get(word)
    if word_id is None and not lower:
        word_id = word2id.get(word.capitalize())
    if word_id is None and not lower:
        word_id = word2id.get(word.title())
    return word_id

def get_wordanalogy_scores(questionanswer_filepath, word2id, embeddings, lower, verbose = False):
    """
    Return (english) word analogy score
    """
    # normalize word embeddings
    row_sums = np.sqrt((embeddings ** 2).sum(1))[:, None]
    rows_with_zerosum = np.argwhere(row_sums == 0)
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(1))[:, None]

    if len(rows_with_zerosum) > 0: # set rows that have been turned to NaN back to 0
        embeddings[rows_with_zerosum[:,0],:] = 0
        print('Fixed NaNs from zero division')

    # scores by category
    scores = defaultdict(dict)

    word_ids = {}
    queries = {}

    num_examples_thrown = 0
    total_examples = 0
    with io.open(questionanswer_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # new line
            line = line.rstrip()
            if lower:
                line = line.lower()

            # new category
            if ":" in line:
                assert line[1] == ' '
                category = line[2:]
                assert category not in scores
                scores[category] = {'n_found': 0, 'n_not_found': 0, 'n_correct': 0}
                word_ids[category] = []
                queries[category] = []
                continue

            # get word IDs
            assert len(line.split()) == 4, line
            word1, word2, word3, word4 = line.split()

            word_id1 = get_word_id(word1, word2id, lower)
            word_id2 = get_word_id(word2, word2id, lower)
            word_id3 = get_word_id(word3, word2id, lower)
            word_id4 = get_word_id(word4, word2id, lower)

            # if at least one word is not found
            if any(x is None for x in [word_id1, word_id2, word_id3, word_id4]):
                scores[category]['n_not_found'] += 1
                num_examples_thrown +=1 
                continue
            else:
                scores[category]['n_found'] += 1
                word_ids[category].append([word_id1, word_id2, word_id3, word_id4])
                # generate query vector and get nearest neighbors
                query = embeddings[word_id1] - embeddings[word_id2] + embeddings[word_id4]
                query = query / np.linalg.norm(query)

                queries[category].append(query)
            total_examples += 1
    if verbose:
        print("Done scanning, threw {} examples out of {} total = {} % discarded".format(num_examples_thrown, total_examples, float(num_examples_thrown)/total_examples * 100))
    
    # Compute score for each category
    total_cats = len(queries)
    curr_cat = 0
    ROW_LIMIT = 500
    
    overall_start_time = time.time()
    with torch.no_grad(): # make sure to not store computational graph info
      for cat in queries:

          start_time = time.time()

          qs_np = np.vstack(queries[cat])
          qs_shape = qs_np.shape
          
          for i in range(0, qs_shape[0], ROW_LIMIT): #allocate matrices of size ROW LIMIT rows 
            if i >= ROW_LIMIT:
              total_cats += 1 
            qs = torch.from_numpy(qs_np[i:i + ROW_LIMIT, :]).cuda()
            keys = torch.from_numpy(embeddings.T).cuda()
            values = qs.mm(keys)

            # free up memory 
            del qs
            del keys 
            torch.cuda.empty_cache()

            word_ids_tensor = torch.tensor(word_ids[cat]).cuda()
            curr_word_ids_tensor = word_ids_tensor[i:i+ROW_LIMIT, :]

            # be sure we do not select input words
            for j, ws in enumerate(curr_word_ids_tensor):
                for wid in [ws[0], ws[1], ws[3]]:
                    values[j, wid] = -1e9
            maxes, indices = values.max(axis = 1)
            correct_indices = curr_word_ids_tensor[:, 2]
            num_correct = torch.sum(torch.eq(indices, correct_indices)).item()
            key = cat + "_{}".format(str(round(i/(ROW_LIMIT))))
            scores[key]['n_correct'] = num_correct

            curr_cat +=1 
            if verbose:
                print('finished batch {} out of {}, took {} seconds'.format(curr_cat, total_cats, time.time() - start_time))
            
            # clean up memory
            del values 
            del word_ids_tensor
            del maxes
            del indices
            del correct_indices 
            torch.cuda.empty_cache()
            # print("Current CUDA snapshot after del and empty cache at end of loop", torch.cuda.memory_allocated())
    
    overall_total_time = time.time() - overall_start_time
    # compute and log accuracies
    
    total_correct = 0
    total_found = 0

    for k in sorted(scores.keys()):
        v = scores[k]
        total_correct += v['n_correct']
        total_found += v.get('n_found', 0)
    if verbose:
        print("total correct: {}, total found: {}".format(total_correct, total_found))
    total_accuracy = float(total_correct)/total_found
    if verbose:
        print("total acc: {}".format(total_accuracy))
    return scores, total_correct, total_found, total_accuracy, overall_total_time

def evaluate(embedding, questionanswer_filepath, word2index):    
    scores, total_correct, total_found, total_accuracy, total_time = get_wordanalogy_scores(questionanswer_filepath, word2index, embedding, True, verbose = False) # False means we have embeddings for uppercased words, True means embeddings for lowercase
    return total_accuracy