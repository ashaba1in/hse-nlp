import numpy as np

from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors
from utils import get_distinct_words


def reduce_to_k_dim(matrix, k=100):
    """ 
    reduce a matrix to dimensionality [matrix.shape[0], k] using TruncatedSVD decomposition

    :args:
        matrix (np.ndarray [vocab_size, vocab_size]): matrix to be reduced
        k (int): desired embedding size
    :return:
        reduced_matrix (np.ndarray [vocab_size, k])
    """
    n_iters = 10

    # your code here

    return reduced_matrix


class BaseEmbeddings(KeyedVectors):
    def __init__(self, corpus, distinct_words=None, word_counter=None, vector_size=100, min_count=10):
        super().__init__(vector_size=vector_size)
        
        self.index_to_key = distinct_words
        self.word_counter = word_counter
        if distinct_words is None or word_counter is None:
            self.index_to_key, self.word_counter = get_distinct_words(corpus, min_count=min_count)
    
        self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}


class CoOccurenceEmbeddings(BaseEmbeddings):
    def __init__(self, corpus, distinct_words=None, window_size=5, vector_size=100, min_count=10):
        super().__init__(corpus, vector_size=vector_size, distinct_words=distinct_words, min_count=min_count)

        self.matrix = self.compute_co_occurrence_matrix(corpus, window_size=window_size)
        self.vectors = reduce_to_k_dim(self.matrix, k=self.vector_size)

    
    def compute_co_occurrence_matrix(self, corpus, window_size=5):
        """
        compute co-occurrence matrix for the given corpus and window_size.

        :args:
            corpus (list of list of strings): corpus of texts
            window_size (int): size of context window
        :return:
            matrix (a symmetric numpy matrix [vocab_size, vocab_size]): co-occurence matrix of word counts. 
            The ordering of the words in the rows/columns should be the same as the ordering of the words given by the get_distinct_words function.
        """

        matrix = np.zeros((len(self), len(self)))

        # your code here
        
        return matrix


class PPMIEmbeddings(CoOccurenceEmbeddings, BaseEmbeddings):
    def __init__(self, corpus, distinct_words=None, window_size=5, vector_size=100, min_count=10):
        BaseEmbeddings.__init__(self, corpus, vector_size=vector_size, distinct_words=distinct_words, min_count=min_count)

        matrix = self.compute_co_occurrence_matrix(corpus, window_size=window_size)
        self.ppmi_matrix = self.compute_ppmi_matrix(matrix)
        self.vectors = reduce_to_k_dim(self.ppmi_matrix, k=self.vector_size)
    
    def compute_ppmi_matrix(self, co_occurrence_matrix):
        """
        compute PPMI matrix using the given co_occurrence_matrix.

        :args:
            co_occurrence_matrix: the output of compute_co_occurrence_matrix function
        :return:
            matrix (a symmetric numpy matrix [vocab_size, vocab_size]): PPMI matrix. 
            The ordering of the words in the rows/columns should be the same as the ordering of the words given by the get_distinct_words function.
            
        note: you should avoid using any loops here
              use natural logarithm
        """

        ppmi_matrix = np.zeros((len(self), len(self)))
        
        # your code here

        return ppmi_matrix


class Word2Vec(BaseEmbeddings):
    def __init__(self, corpus, distinct_words=None, vector_size=100, window_size=5,
                 min_count=10, batch_size=None, n_negative=5, n_epoches=5):
        super().__init__(corpus, vector_size=vector_size, distinct_words=distinct_words, min_count=min_count)

        self.corpus = corpus
        self.window_size = window_size
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = np.max([len(text) for text in corpus])
        self.n_negative = n_negative
        self.alpha = 0.001
        
        self.center_W = ... # your code here
        self.context_W = ... # your code here
        
        self.train(n_epoches)
        self.vectors = self.center_W
        
        # your code here

    def train(self, n_epoches=5):
        """
        trains self.center_W and self.context_W matrices
        """
        # your code here
        pass