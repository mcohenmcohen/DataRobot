import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import gensim
from itertools import combinations
import time
import operator
import os
import matplotlib
import matplotlib.pyplot as plt
import nlp_utils


class NMF_TopicModeller(object):
    '''
    Wrapper for NMF
    '''
    def __init__(self, k_topics=8,
                 max_vocab_size=5000, min_df=20, max_df=1.0, ngram_range=(1, 1),
                 **kwargs):
        '''
        Input:
            max_vocab_size - upper bound limit to the number of features/terms
            min_df - vectorizer: min document frequency
            max_df - vectorizer: ignore words with a document frequency above %
            ngram_range - number of ngrams to search for, starting from 1.
                          The lower and upper boundary of the range of n-values
                          for different n-grams to be extracted
        '''
        # self.x_train = X_train  # document set as Pandas Series
        self.document_term_mat = None
        self.model = None
        self.W = None  # Populated only for NMF model
        self.H = None  # Populated only for NMF model

        token_pattern = nlp_utils.get_token_pattern()
        stop_words = nlp_utils.get_stop_words()

        self.vectorizer = TfidfVectorizer(token_pattern=token_pattern,
                                          min_df=min_df,
                                          max_df=max_df,
                                          max_features=max_vocab_size,
                                          stop_words=stop_words,
                                          ngram_range=ngram_range)

    def vectorize(self, docs):
        '''
        Vectorize the document content and fit the NMF

        Input
            train_docs - Training document set
        Output:
            the fit model
        '''
        # list of document content
        # eg, resume content for each user or job posting description content
        print('Number of documents to process: %s\n' % docs.shape)

        print("Extracting Vectorizer features...")
        t1 = time.time()
        self.document_term_mat = self.vectorizer.fit_transform(docs)
        print("- Time: %0.3fs.\n" % (time.time() - t1))

    def fit(self, docs, k_topics):
        '''
        Input
            docs - Documents to topic model
            k_topics - k number of topics to generate
        '''
        if self.document_term_mat is None:
            print('Vectorizer wasn\'t fitted.  '
                  'Call your TopicModeller.vectorize first.')
            return

        print("Fitting model with %d documents.  "
              "Vectorizer: \n%s" % (docs.shape[0], self.vectorizer))

        self.model = NMF(n_components=k_topics,
                         alpha=.1, l1_ratio=.5, init='nndsvd')

        t1 = time.time()
        W = self.model.fit_transform(self.document_term_mat)
        H = self.model.components_
        self.W = W
        self.H = H
        print("- Time: %0.3fs.\n" % (time.time() - t1))

        # self.describe_matrix_factorization_results(self.document_term_mat, W, H, n_top_words=20)

    def document_term_mat_toframe(self):
        all_feature_names = self.vectorizer.get_feature_names()
        dtm = self.document_term_mat.todense()
        dfv = pd.DataFrame(dtm, columns=all_feature_names)
        return dfv

    def reconst_mse(self, target, left, right):
        '''
        Calcuate the mean squared error between soruce matrix and
        the reconstruction of the matrix with W*H
        '''
        return (np.array(target - left.dot(right))**2).mean()

    def describe_matrix_factorization_results(self, document_term_mat, W, H,
                                              n_top_words=15):
        '''
        For each latent topic print the top n words assocaited with that topic

        TODO: print probabilities
        '''
        feature_words = self.vectorizer.get_feature_names()
        print("Reconstruction mse: %f" % (self.reconst_mse(document_term_mat,
                                                           W, H)))
        for topic_num, topic in enumerate(H):
            top_features = ', '.join([feature_words[i]
                                     for i in topic.argsort()[:-n_top_words-1:-1]])
            print("Topic %d: %s\n" % (topic_num, top_features))

        return

    def rank_terms(self):
        # get the sums over each column/term
        sums = self.document_term_mat.sum(axis=0)
        terms = self.vectorizer.get_feature_names()
        # map weights to the terms
        weights = {}
        for col, term in enumerate(terms):
            weights[term] = sums[0, col]
        # rank the terms by their weight over all documents
        return sorted(list(weights.items()), key=operator.itemgetter(1), reverse=True)

    def get_doc_terms_and_scores(self, doc_index):
        '''
        Return the tfidf values for vectorized terms for a document.

        Input:
            doc_index:  The row index number for the fitted document_term_matrix
        Output:
            dictionary of {doc terms: tfidf scores}

        Hint: A sorted print to use:
            for key, value in sorted(tfidf_scores.iteritems(),
                                     key=lambda (k,v): (v,k), reverse=True):
                print "{:<10}: {:<10}".format(key, value)

        '''
        all_feature_names = self.vectorizer.get_feature_names()
        dtm = self.document_term_mat.todense()

        doc_terms_indicies = dtm[doc_index, :].nonzero()[1]
        tfidf_scores = {all_feature_names[term_idx]: dtm[doc_index, term_idx]
                        for term_idx in doc_terms_indicies}

        return tfidf_scores

    def print_W_probs(self, W):
        '''
        Input
            W NMF matrix
        '''
        probs = (W / W.sum(axis=1, keepdims=True)).flatten()
        ordered = np.argsort(probs)[::-1]
        for idx in ordered:
            print('Topic %s: %0.3f' % (idx, probs[idx]))

    def get_normalized_probs(self, topic_weights):
        '''
        Return the normalized topic cluseter weights for a given row vector
        '''
        topic_weights = topic_weights.flatten()
        probs = (topic_weights / topic_weights.sum())
        return probs

    def get_top_topics_and_topic_probs(self):
        '''
        Generate the probability of each topic for each row (eg, job posting)
        in W, and add the top topic and probability and return each as a list,
        (for example to be used as new columns added to a dataframe)
        '''
        # For each row, get the topic weights, normalize, order by
        # weight value, and store in a list to add to the dataframe
        top_topics = []
        top_topic_weights = []
        for row_idx in range(self.W.shape[0]):
            W = self.W[row_idx]
            probs = self.get_normalized_probs(W)

            ordered_idxs = np.argsort(probs)[::-1]
            top_topics.append(ordered_idxs[0])
            top_topic_weights.append(probs[ordered_idxs[0]])

        return (top_topics, top_topic_weights)

    def classify_new_doc(self, doc, display=True):
        '''
        Classify a new document using the fit NMF model.

        Input
            doc - string
        Output
            Dictionary of topics and their weights
            Optional output on (True) by default
        '''
        if not self.model:
            'A model has not been fit yet.'
        if type(doc) != str:
            'Input document must be a string'

        # Using NMF
        # TODO word2Vec
        document_term_mat = self.vectorizer.transform([doc])
        W = self.model.transform(document_term_mat)
        # H = self.model.components_

        probs = (W / W.sum(axis=1, keepdims=True)).flatten()
        ordered = np.argsort(probs)[::-1]
        topic_dict = {}
        for idx in ordered:
            topic_dict[idx] = probs[idx]
            if display:
                print('Topic %s: %0.3f' % (idx, probs[idx]))

        return topic_dict


def plot_optimal_k(docs, document_term_mat, vectorizer,
                   kmin=3, kmax=15, num_top_terms=15,
                   alpha=.1, l1_ratio=.5,
                   dim_size=500, min_df=20, max_vocab_size=5000,
                   model_file_path='./data/',
                   model_file_name='w2v-model.bin'):
    '''
    Run NMF for each k between min and max and plot to assess optimal k.

    Input
        docs - corpus of docuemnts as a list
        document_term_mat - TFIDF matrix from the vectorizer
        vectorizer - scikit-learn TFIDF vectorizer (trained in TopicModeller)

    Returns:
        Int - optimal k number
    '''
    topic_models = []

    # Run NMF for each value of k
    for k in range(kmin, kmax+1):
        t1 = time.time()

        # Run NMF
        model = NMF(n_components=k, init='nndsvd',
                    alpha=alpha, l1_ratio=l1_ratio)

        W = model.fit_transform(document_term_mat)
        H = model.components_

        # Store for iterating over all the models (of each k size)
        topic_models.append((k, W, H))

        print("Processed NMF for k=%d of %d - Time: %0.3fs." % (k, kmax, (time.time() - t1)), end='\r', flush=True)
    print()

    # If the model is already built get it from disk, otherwise
    # build a Skipgram Word2Vec model from all documents
    # in the input file using Gensim:
    model_path = model_file_path + model_file_name
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    w2v_model = None
    try:
        w2v_model = gensim.models.Word2Vec.load(model_path)
    except Exception as e:
        print('No existing word2vec model found to load. Exception: %s.\n'
              'Building it...' % (e))

    # w2v_model = None - uncomment to force rebuild every time
    if w2v_model:
        print('Existing word2vec Model loaded from \'%s\'' % model_path)
    else:
        docgen = nlp_utils.TokenGenerator(docs)
        # Process w2v with model of n dimensions and min doc-term freq as min_df
        t1 = time.time()
        w2v_model = gensim.models.Word2Vec(docgen, sg=1, size=dim_size,
                                           max_vocab_size=max_vocab_size,
                                           min_count=min_df)
        print("- Time: %0.3fs." % (time.time() - t1))
        # Save for later use, so that we do not need to rebuild it:
        print('Saving it...')
        w2v_model.save(model_path)

    print(('word2vec model has %d terms' % len(w2v_model.wv.vocab)))

    # Implement TC-W2V coherence score measure
    def calculate_coherence(w2v_model, term_rankings):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # check each pair of terms
            pair_scores = []
            # print 'Topic %s: %s top words: %s' % (topic_index,
            #                                       len(term_rankings[topic_index]),
            #                                       term_rankings[topic_index])
            for pair in combinations(term_rankings[topic_index], 2):
                pair_scores.append(w2v_model.similarity(pair[0], pair[1]))
            # get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # get the mean score across all topics
        return overall_coherence / len(term_rankings)

    # Function to get the topic descriptor
    # (i.e. list of top terms) for each topic:
    def get_descriptor(all_terms, H, topic_index, num_top_terms):
        # reverse sort the values to sort the indices
        top_indices = np.argsort(H[topic_index, :])[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:num_top_terms]:
            top_terms.append(all_terms[term_index])
        return top_terms

    # Process each of the models for different values of k:
    vocab = vectorizer.get_feature_names()
    # vocab = w2v_model.wv.vocab

    # Process each of the models for different values of k:
    k_values = []
    coherences = []
    print('Calculating coherence scores...')
    for (k, W, H) in topic_models:
        # Get all topic descriptors - the term_rankings, based on top n terms
        term_rankings = []
        for topic_index in range(k):
            # term_rankings.append(get_descriptor(vocab, H, topic_index, num_top_terms))
            top_words = [vocab[i] for i in H[topic_index, :].argsort()[:-num_top_terms - 1:-1]]
            top_words = [x for x in top_words if x in w2v_model.wv.vocab]
            term_rankings.append(top_words)
        # Calculate the coherence based on our Word2vec model
        k_values.append(k)
        coherences.append(calculate_coherence(w2v_model, term_rankings))
        # print(('K=%02d: Coherence=%.4f' % (k, coherences[-1])))

    # Plot a line of coherence scores to identify an appropriate k value.
    plt.style.use("ggplot")
    matplotlib.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(13, 7))
    # Create the line plot
    ax = plt.plot(k_values, coherences)
    plt.xticks(k_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Mean Coherence")
    # Add the points
    plt.scatter(k_values, coherences, s=120)
    # Find and annotate the maximum point on the plot
    ymax = max(coherences)
    xpos = coherences.index(ymax)
    best_k = k_values[xpos]
    plt.annotate('k=%d' % best_k, xy=(best_k, ymax), xytext=(best_k, ymax),
                 textcoords="offset points", fontsize=16)
    print('Optimal number of k topics: %s' % best_k)
    # Show the plot
    plt.show()

    k = best_k
    # Get the model that we generated earlier.
    W = topic_models[k-kmin][1]
    H = topic_models[k-kmin][2]

    # Display the topics and descriptor words for the best k model
    for topic_index in range(k):
        descriptor = get_descriptor(vectorizer.get_feature_names(),
                                    H, topic_index, num_top_terms)
        str_descriptor = ", ".join(descriptor)
        print(("Topic %02d: %s" % (topic_index, str_descriptor)))

    return int(k)
