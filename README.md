Code for our UCL NLP Final Project, using Hoyer-Square regularisation for learning sparse embeddings.

William Lamb, Kush Madlani, Udeepa Meepegama, Aneesh Pappu

## Directory Structure
### Code 
Code for training the WAKU pipeline, downstream task evaluation, etc. resides in the main directory titled "waku".

### Data
All input data should be placed in the folder titled "raw_data". Trained embeddings should be written to "embeddings".

The "questions-words" dataset for the word analogy task can be found [here](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt)

### Downstream Evaluation
A notebook which loads trained embeddings, evaluates them on the sentiment analysis, word analogy, word similarity, and word intrusion tasks, and produces t-SNE visualisations can be found in the folder titled "notebooks".
