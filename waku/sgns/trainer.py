import torch
import torch.optim as optim
import numpy as np
import time

from waku.sgns.model import SkipGramModel
from waku.sgns.dataloader import DataPipeline

class Word2Vec:
    """
    Wrapper for skip gram word2vec model with:
        1. Negative sampling.
        2. Sub-sampling of frequent words.
    """
    def __init__(self, center_words, context_words, neg_samples,
                 word2index, index2word, word_freq, emb_dimension=300,
                 verbose=False):
        """
        Arguments:
        -----------
        center_words : `numpy.ndarray`
            List of center words from the data.
        context_words : `numpy.ndarray`
            (len(center_words)) List of corresponding context words.
        neg_samples : `numpy.ndarray`
            (len(center_words), num_neg_samples) Matrix of negative samples 
            for all center words. 
        word_freq  : `dict`
            Dictionary of frequency of each word in corpus.
        word2index : `dict`, {word_str: word_index}
            Mapping from word to index.
        index2word : `dict`, {word_index: word_str}
            Mapping from index to word.                      
        emb_dimension : `int`
            Embedding dimension - the size of each embedding vector (50-500).
        verbose : `Boolean`
            If true print statistics.                                  
        """          
        self.word2index = word2index
        self.index2word = index2word
        self.word_freq = word_freq
        # Get DataLoader that returns batches of data
        self.pipeline = DataPipeline(center_words, context_words, neg_samples)
        # Get number of unique words in corpus - vocabulary size
        self.vocab_size = len(word2index.keys()) 
        # Check if GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialise skip gram model
        self.emb_dimension = emb_dimension
        self.model: SkipGramModel = SkipGramModel(self.vocab_size, emb_dimension).to(self.device) 
        # Print
        self.verbose = verbose
        if verbose:
            print("Device used: ", self.device)        
            print("Length of corpus: {:,}".format(sum(word_freq.values())))
            print("Vocabulary size: {:,}".format(len(word_freq.keys()))) 
            print("Number of training examples: {:,}".format(len(center_words)))        

    def train(self, 
              regulariser=None,
              decay=0,
              l2_decay=0,
              prune_thresh=0,
              epochs=10, 
              batch_size=128, 
              learning_rate=1e-3,
              solver="sgd",
              save_checkpoint_epoch=0,
              save_checkpoint_path=None,
              load_checkpoint_path=None,
              time_limit=169200,
              freeze=False):
        """
        Train Skip Gram word2vec model.

        Parameters:
        -----------
        regulariser : `str`
            The regulariser to use in the loss function.  
        decay : `float`
            The coefficient of the regulariser term.   
        l2_decay : `float`
            Whether to use L2 regulariser. 
        prune_thresh : `float`
            The threshold for pruning.  
        epochs : `int`
            Number of epochs for training.   
        batch_size : `int`
            Number of word pairs in each batch.    
        learning_rate : `float`
            The learning rate for backpropogation.  
        solver : `str`
            The optimiser to use, either `sparse_adam` or `sgd` or `adam`.
        save_checkpoint_epoch : `int`
            Save checkpoint every `save_checkpoint_epoch`.
        save_checkpoint_path : 'Nonetype` or `str`
            The path to save the checkpoint of the model.
        load_checkpoint_path : `NoneType` or `str`
            Path to load checkpoint.
        time_limit : `int`
            The time limit for the stopping condition.
        freeze : `Boolean`
            Whether to freeze the zero elements of the input embedding matrix 
            during training.

        Returns:
        --------
        Train_loss : `list` of `float`
            The training loss per example in each epoch.
        """
        if self.verbose:
            print("Embedding dimension: ", self.emb_dimension)
            print("Regulariser:         ", regulariser)
            print("Lambda:              ", decay)
        self.regulariser = regulariser
        # Set model to train mode
        self.model.train()
        # Get optimiser for training
        if solver.lower() == "sgd":
            self.optimiser = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=l2_decay)
        elif solver.lower() == "adam": 
            self.optimiser = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_decay)

        # Load from checkpoint
        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)
            
        # Get mask for zeroing gradients    
        if freeze:
            mask = (self.model.input_embeddings.weight.data == 0)            
        
        # Time after first epoch
        break_time = time.time()
        # Measures
        Train_loss = list()
        # Iterate through epochs
        for epoch in range(epochs):
            start_time = time.time()
            # Loss in epoch
            train_loss = 0.0
            # Get batches of center words, context words and negative samples
            for batch_i, (batch_inputs, batch_labels, batch_negs) in enumerate(self.pipeline.generate_batch(batch_size)):
                
                # Push onto GPU if available
                batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).to(self.device) 
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device) 
                batch_negs = torch.tensor(batch_negs, dtype=torch.long).to(self.device) 

                # Zero accumulated gradients
                self.model.zero_grad()
                # Calculate loss (skip gram negative sample loss) by running through model
                # loss per training example
                loss = self.model(batch_inputs, batch_labels, batch_negs)

                # Apply regulariser
                reg = 0.0
                if decay > 0 and self.regulariser !="vanilla":
                    for param_i, (param_name, param) in enumerate(self.model.named_parameters()):
                        # Should only regularise rows effected in the embeddings by the batch
                        if param_name == "input_embeddings.weight":
                            # Input embeddings
                            rows_effected = torch.unique(batch_inputs)   
                        elif param_name == "output_embeddings.weight":
                            # Output embeddings
                            unique_labels = torch.unique(batch_labels) 
                            unique_negs = torch.unique(batch_negs) 
                            rows_effected = torch.unique(torch.cat([unique_labels, unique_negs],dim=0))
                        param = param[rows_effected] 
                        # get regulairisation term per example
                        if param.requires_grad and torch.sum(torch.abs(param))>0:
                            if self.regulariser == "hs":
                                reg += (torch.sum(torch.abs(param))**2)/torch.sum(param**2) / rows_effected.size()[0]
                            elif self.regulariser == "hoyer":
                                reg += torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2)) / rows_effected.size()[0]
                            elif self.regulariser == "l1":
                                reg += torch.sum(torch.abs(param)) / rows_effected.size()[0]
                            elif self.regulariser == "transformed_l1":
                                reg += torch.sum(2*torch.abs(param)/(1+torch.abs(param))) / rows_effected.size()[0]
                # Get total loss
                total_loss = loss + decay*reg
                # Backpropogation: calculating gradients
                total_loss.backward()
                
                if freeze:
                    # Zero gradients of zero elements                
                    self.model.input_embeddings.weight.grad[mask] = 0
                    
                # Update weights
                self.optimiser.step()

                # Add to train loss of epoch
                train_loss += total_loss.item() * batch_inputs.numel()

            # After epoch, set data index to zero
            self.pipeline.data_index = 0
            # Save loss for epoch 
            train_loss /= len(self.pipeline.center_words)
            Train_loss.append(train_loss)
            # Print statistics for epoch
            if self.verbose:
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                    "Train Loss: {:.4f}...".format(train_loss),
                    "Time Taken: {:,.4f} seconds".format(time.time()-start_time))
            
            # Save model after every `save_checkpoint_epoch` epochs
            if save_checkpoint_path is not None and epoch % save_checkpoint_epoch == 0 and epoch !=0:
                self.save_checkpoint(save_checkpoint_path+"_"+str(epoch))            

            # Early stopping using time
            if time.time() - break_time >= time_limit:
                print("Time Taken: {:,.4f} seconds".format(time.time()-break_time))
                break

        # Prune the matrix after training
        if prune_thresh != 0:
            self.prune(prune_thresh) 

        return Train_loss

    def test(self, center_words, context_words, neg_samples, batch_size):
        """
        Test Skip Gram word2vec model.

        Parameters:
        -----------
        center_words : `numpy.ndarray`
            List of center words from the data.
        context_words : `numpy.ndarray`
            (len(center_words)) List of corresponding context words.
        neg_samples : `numpy.ndarray`
            (len(center_words), num_neg_samples) Matrix of negative samples 
            for all center words. 
        batch_size : `int`
            Number of word pairs in each batch.    

        Returns:
        --------
        test_loss : `float`
            The test loss per example for over the data.
        """
        if self.verbose:
            print("Number of test examples: {:,}".format(len(center_words)))   
        # Get DataLoader that returns batches of data
        test_pipeline = DataPipeline(center_words, context_words, neg_samples)
        # Set to evaluate mode
        self.model.eval()  
        # Loss
        test_loss = 0
        # Get batches of center words, context words and negative samples
        for batch_i, (batch_inputs, batch_labels, batch_neg) in enumerate(test_pipeline.generate_batch(batch_size)):
            
            # Push onto GPU if available
            batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).to(self.device) 
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device) 
            batch_neg = torch.tensor(batch_neg, dtype=torch.long).to(self.device)       

            # Calculate loss (skip gram negative sample loss) by running through model
            # loss per training example
            loss = self.model(batch_inputs, batch_labels, batch_neg)    

            # Add to train loss of epoch
            test_loss += loss.item()   

        return test_loss #/ len(center_words)  

    def prune(self, prune_thresh):
        """
        Prune the embeddings matrix.
        If the absolute values of the elements are smaller than the 
        prune threshold then set them to zero.        

        Parameters:
        -----------
        prune_thresh : `float`
            The threshold for pruning.         
        """
        self.model.input_embeddings.weight.data[torch.abs(self.model.input_embeddings.weight.data) < prune_thresh] =  0

    # Helper Functions
    #-----------------
    def save_checkpoint(self, path):
        """
        Save checkpoint of model

        Parameters:
        -----------
        path : `str`
            The path to save the checkpoints.
        """
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimiser_state_dict': self.optimiser.state_dict()}, 
                   path+'_checkpoint.pth.tar')     

    def load_checkpoint(self, path):
        """
        Save checkpoint of model.

        Parameters:
        -----------
        path : `str`
            The path to save the checkpoints.
        """
        checkpoint = torch.load(path+'_checkpoint.pth.tar')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    def save_model(self, out_path):
        """
        Save model.

        Parameters:
        -----------
        out_path : `str`
            The output folder for saving the model.
        """
        torch.save(self.model.state_dict(), out_path+'_model.pt')

    def load_model(self, model_path):
        """
        Load model.

        Parameters:
        -----------
        mode_path : `str`
            The path where the model is saved.
        """        
        self.model.load_state_dict(torch.load(model_path))    

    def get_embeddings(self):
        """
        Return the embeddings matrix.

        Returns:
        --------
        embeddings : `numpy.ndarray`
            (vocab_size, emb_dimensions) The embeddings matrix.
        """
        return self.model.input_embeddings.weight.cpu().data.numpy()

    def load_embeddings(self, embeddings):
        """
        Replace current input embedding with `embeddings`
        
        Parameters:
        -----------
        embeddings : `numpy.ndarray`
            (vocab_size, emb_dimensions) The embeddings matrix.
        """
        self.model.input_embeddings.weight.data = torch.tensor(embeddings,
                                                               dtype=torch.float).to(self.device)
        
    def save_emb2txt(self, filename):
        """
        Save embedding matrix to a .txt file.
        The first column are the word strings and the following columns
        are the embeddings of the words.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the embeddings matrix.
        """
        embeddings = self.get_embeddings()
        with open(filename, 'w') as f:
            for index, word in self.index2word.items():
                e = ' '.join(map(lambda x: str(x), embeddings[index]))
                f.write('%s %s\n' % (word, e))

    def save_emb2npz(self, filename):
        """
        Save embedding matrix to a .npz (compressed) file.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the embeddings matrix.
        """
        embeddings = self.get_embeddings()
        np.savez_compressed(filename, a=embeddings)        
    
    def get_embedding(self, input):
        """
        Get the embedding of input

        Parameters:
        -----------
        input : `str` or `int`
            A word or the index of the word.

        Returns:
        --------
        emb : `torch.tensor`   
            The embedding of the input.
        """
        if type(input) is str:
            input = torch.tensor(self.word2index[input], dtype=torch.long).to(self.device).unsqueeze(0)
        return self.model.predict(input)

    def most_similar(self, word, top_k=8):
        """
        Get top k most similar words to current using cosine similarity.

        Parameters:
        -----------
        word : `str`
            The word to find the top k most similar words of.
        top_k : `int`
            The number of words to find around `word`.

        Returns:
        --------
        top_list : `list` of `str`
            A list of the top k most similar words to `word` from embedding.
        """
        magnitudes = self.model.input_embeddings.weight.pow(2).sum(dim=1).sqrt().unsqueeze(0)
        # Get the embedding of the word
        emb = self.get_embedding(word)
        # Similarity measure of word with other words
        sim = torch.mm(emb, self.model.input_embeddings.weight.transpose(0, 1)) / magnitudes
        # Get top k similar
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        # Convert index to word
        top_list = list()
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list