import numpy as np
import torch
import torch.optim as optim

import time 
import pickle

class Word2Vec:
    """
    Wrapper for skip gram word2vec model with:
        1. Negative sampling.
        2. Sub-sampling of frequent words.
    """
    def __init__(self, input_filename, emb_dimension, thresh_subsample=1e-4):
        """
        Arguments:
        -----------
        input_filename : `str`
            The input filename for the data.
        emb_dimension : `int`
            Embedding dimension - the size of each embedding vector (50-500).  
        thresh_subsample : `float`
            Subsampling threshold.
        """
        # Get data, vocabulary size, word frequency, word2index and index2word mappings
        self.data, self.vocab_size, self.word_freq, self.word2index, self.index2word = build_dataset(read_data(input_filename))
        print("Length of corpus before subsampling: {:,}".format(len(self.data)))
        # List of all unique indices used to map words to indices
        self.vocabs = list(set(self.data))
        # Sub-sampling of frequent words
        self.data, self.subsampled_word_freq = subsample(self.data, self.word_freq, thresh_subsample)
        print("Length of corpus after subsampling: {:,}".format(len(self.data)))
        # Check if GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used: ", self.device)
        # Initialise skip gram model
        self.model: SkipGramModel = SkipGramModel(self.vocab_size, emb_dimension).to(self.device) 
        # The index of the data the generator of the batch has got to
        self.data_index = 0

    def train(self, 
              regulariser=None,
              decay=0,
              window_size =5,
              num_neg_samples=5, 
              epochs=10, 
              batch_size=128, 
              learning_rate=1e-3,
              solver="sgd",
              table_size=1e8,
              checkpoint_path=None,
              load_from_checkpoint=False):
        """
        Train Skip Gram word2vec model.

        Parameters:
        -----------
        regulariser : `NoneType` or `str`
            The regulariser to use in the loss function.  
        decay : `float`
            The coefficient of the regulariser term.      
        window_size : `int`
            Max skip length between words.     
        num_neg_samples : `int`
            Number of negative samples.
        epochs : `int`
            Number of epochs for training.   
        batch_size : `int`
            Number of word pairs in each batch.    
        learning_rate : `float`
            The learning rate for backpropogation.  
        solver : `str`
            The optimiser to use, either `sparse_adam` or `sgd`.
        table_size : `int`
            Size of the table for generating negative samples.
        checkpoint_path : 'Nonetype` or `str`
            The path to save the checkpoint of the model.
        load_from_checkpoint : `Boolean`
            Whether to load model etc. from a checkpoint.

        Returns:
        --------
        Train_loss : `list` of `float`
            The training loss at each epoch.
        """
        # Set model to train mode
        self.regulariser = regulariser
        self.model.train()
        # Get optimiser for training
        if solver.lower() == "sgd":
            optimiser = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif solver.lower() == "sparse_adam": 
            optimiser = optim.SparseAdam(self.model.parameters(), lr=learning_rate)

        # Load from checkpoint
        if load_from_checkpoint:
            epoch_start, optimiser, saved_train_loss = self.load_checkpoint(optimiser, checkpoint_path)
        else:
            epoch_start = 0 

        # Create data class
        pipeline = DataPipeline(self.data, self.vocabs, self.subsampled_word_freq, table_size)
        # Measures
        Train_loss = list()
        for epoch in range(epoch_start, epochs):
            start_time = time.time()
            batch100_time = time.time()
            if load_from_checkpoint and epoch == start_epoch:
                train_loss = saved_train_loss
            else:
                train_loss = 0

            # Get center words and corresponding context words
            for batch_i, (batch_inputs, batch_labels, data_index) in enumerate(pipeline.generate_batch(batch_size, window_size, self.data_index)):
                self.data_index = data_index
                # Get negatively sampled words for center words
                batch_neg = pipeline.get_neg_data(batch_inputs.shape[0], num_neg_samples, batch_inputs)
                
                # Push onto GPU if available
                batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).to(self.device) 
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device) 
                batch_neg = torch.tensor(batch_neg, dtype=torch.long).to(self.device) 

                # Zero accumulated gradients
                self.model.zero_grad()
                # Calculate loss (skip gram negative sample loss) by running through model
                loss = self.model(batch_inputs, batch_labels, batch_neg)

                # Apply regulariser
                reg = 0.0
                if decay and self.regulariser is not None:
                    for param in self.model.parameters():
                        if param.requires_grad and torch.sum(torch.abs(param))>0:
                            if self.regulariser == "hs":
                                reg += (torch.sum(torch.abs(param))**2)/torch.sum(param**2)
                            elif self.regulariser == "hoyer":
                                reg += torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2))
                            elif self.regulariser == "l1":
                                reg += torch.sum(torch.abs(param))
                            elif self.regulariser == "transformed_l1":
                                reg += torch.sum(2*torch.abs(param)/(1+torch.abs(param)))

                # Get total loss
                total_loss = loss + decay*reg
                # Backpropogation: calculating gradients
                total_loss.backward()
                # Update weights
                optimiser.step()

                # Add to train loss of epoch
                train_loss += total_loss.item() * batch_inputs.size()[0]
                
                # Print time for 100 batches
                if batch_i % 100 == 0 and batch_i !=0:
                    print("Batch: {}...".format(batch_i+1),
                          "Cumulative Train Loss: {:.4f}...".format(train_loss / (batch_i*batch_size + batch_inputs.size()[0])),
                          "Time Taken: {:,.4f} seconds".format(time.time()-batch100_time))
                    batch100_time = time.time()

                # Save model after every 3000 batches
                if batch_i % 3000 == 0 and batch_i !=0:
                    self.save_checkpoint(epoch, train_loss, optimiser, data_index, checkpoint_path)

            # After iterating through all data, set data index to zero
            self.data_index = 0
            # Save loss for epoch 
            train_loss /= (batch_i*batch_size + batch_inputs.size()[0])
            Train_loss.append(train_loss)
            # Print statistics for epoch
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Train Loss: {:.4f}...".format(train_loss),
                  "Time Taken: {:,.4f} seconds".format(time.time()-start_time))

        return Train_loss

    # Helper Functions
    #-----------------
    def save_checkpoint(self, epoch, loss, optimiser, data_index, path):
        """
        Save checkpoint of model after every 10000 batches.

        Parameters:
        -----------
        epoch : `int`
            The epoch at which model is being saved at.
        loss : `float` 
            The loss accumualted so far.
        optimiser : `torch.optim`
            The optimiser used for training.
        data_index : `int`
            The index of the data the batch should start from.
        path : `str`
            The path to save the checkpoints.
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'loss': loss,
                    'data_index': data_index}, 
                   path+'checkpoint.pth.tar')     

    def load_checkpoint(self, optimiser, path):
        """
        Save checkpoint of model.

        Parameters:
        -----------
        optimiser: `torch.optim`
            Optimiser used for training.
        path : `str`
            The path to save the checkpoints.

        Returns:
        --------
        epoch : `int`
            The epoch at which model was last saved. 
        optimiser : `torch.optim`
            Optimiser used for training
        loss : `float`
            The loss from training so far.
        """
        checkpoint = torch.load(path+'checkpoint.pth.tar')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.data_index = checkpoint['data_index']
        return epoch, optimiser, loss

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
        embeddings = self.model.input_embeddings.weight.cpu().data.numpy()
        with open(filename, 'w') as f:
            for index, word in self.index2word.items():
                e = ' '.join(map(lambda x: str(x), embeddings[index]))
                f.write('%s %s\n' % (word, e))

    def save_emb2npy(self, filename):
        """
        Save embedding matrix to a .npy file.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the embeddings matrix.
        """
        embeddings = self.model.input_embeddings.weight.cpu().data.numpy()
        np.save(filename, embeddings)

    def save_word2index(self, filename):
        """
        Pickle word2index dictionary.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the word2index dictionary.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.word2index, f)       

    def save_subsampled_word_freq(self, filename):
        """
        Pickle subsampled_word_freq dictionary.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the subsampled word frequency dictionary.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.subsampled_word_freq, f)             

    def save_model(self, out_path):
        """
        Save model.

        Parameters:
        -----------
        out_path : `str`
            The output folder for saving the model.
        """
        torch.save(self.model.state_dict(), out_path + '/model.pt')

    def load_model(self, model_path):
        """
        Load model.

        Parameters:
        -----------
        mode_path : `str`
            The path where the model is saved.
        """        
        self.model.load_state_dict(torch.load(model_path))        

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
            input = self.word2index[input]
        return self.model.predict(input)

    def most_similar(self, word, top_k=8):
        """
        Get top k most similar words to current.

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
        index = self.word2index[word]
        index = torch.tensor(index, dtype=torch.long).to(self.device).unsqueeze(0)
        emb = self.model.predict(index)
        sim = torch.mm(emb, self.model.input_embeddings.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list