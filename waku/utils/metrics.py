#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:59:33 2020

@author: udeepa
"""
import numpy as np

def measure_sparsity(embeddings):
    """
    Function to measure sparisty
    sparsity = (# of zeros elements in matrix) / (# of elements in matrix)

    Parameters:
    -----------
    embeddings : `numpy.ndarray`
        (vocab_size, emb_dimensions) The embeddings matrix.

    Returns:
    --------
    sparsity : `float`
        Measure of sparsity.
    """
    return (embeddings.size - np.count_nonzero(embeddings)) / embeddings.size