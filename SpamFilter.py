import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bag_of_words import text_to_words

SPAM = "spam"
HAM = "ham"

class SpamFilter(object):
	def __init__(self):
		self.vectorizer = CountVectorizer(analyzer="word",   \
                                     	  tokenizer=None,    \
                                          preprocessor=None, \
                                          stop_words=None,   \
                                          max_features=5000, \
                                          binary=True) 
		self.prior_spam = 0.5
		self.prior_ham = 0.5
		self.posterior_spam = []
		self.posterior_ham = []

	def train(self, train_filename):
		# Preprocess the train data
		train_data = pd.read_csv(train_filename, encoding='latin-1')

		train_data = train_data.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
		for _, data in train_data.iterrows():
			data.ix[1] = text_to_words(data.ix[1])

		# Vectorize the train data
		train_data_features = self.vectorizer.fit_transform(train_data.ix[:,1].tolist()).toarray()
		
		# Group by label
		spam = []
		ham = []
		for index, data in train_data.iterrows():
			if data.ix[0] == "spam":
				spam.append(train_data_features[index])
			else:
				ham.append(train_data_features[index])
		spam = np.array(spam)
		ham = np.array(ham)

		# Calculate the likelihood for each word in both spam/ham group
		num_spam = spam.shape[0]
		num_ham = ham.shape[0]
		if num_spam == 0 or num_ham == 0:
			print("This is not a good dataset")
			return
		
		count_spam = np.sum(spam, axis=0)
		count_ham = np.sum(ham, axis=0)
		
		occur = count_spam + count_ham	# occurences of the word in learning phase 

		prob_spam = count_spam / num_spam
		prob_ham = count_ham / num_ham
		prob_spam[prob_spam == 0] = 1 / num_spam
		prob_ham[prob_ham == 0] = 1 / num_ham

		# Calculate the posterior probability for the group given each word
		self.posterior_spam = (prob_spam * self.prior_spam) / (prob_spam * self.prior_spam + prob_ham * self.prior_ham)

		# Use corrected posterior probability (for the words were encountered only a few times)
		s = 3	# strength we give to background information about incoming spam 
		self.posterior_spam = (s * self.prior_spam + occur * self.posterior_spam) / (s + occur)

	def query(self, query_filename):
		query_data = pd.read_csv(query_filename, encoding='latin-1')
		
		res = np.array([])
		for _, q in query_data.iterrows():
			predictedVal = self.predict(q.ix[0])
			res = np.append(res, predictedVal)

		return res

	def predict(self, query_text):
		# Preprocess the query data
		query_text = text_to_words(query_text)

		query_words = query_text.split(" ")
		query_vocab = {}
		
		train_vocab = self.vectorizer.get_feature_names()
		
		for word in query_words:
			if word not in query_vocab:
				query_vocab[word] = self.posterior_spam[train_vocab.index(word)] if word in train_vocab else 0.4
		
		sorted_query_vocab = sorted(query_vocab.values(), reverse=True)
		
		high_posteriors = np.array(sorted_query_vocab[:15]) if len(sorted_query_vocab) >= 15 else np.array(sorted_query_vocab)

		# Calculate the joint probability
		high_posteriors_opposite = 1 - high_posteriors
		joint_prob = np.prod(high_posteriors) / (np.prod(high_posteriors) + np.prod(high_posteriors_opposite))
		
		return SPAM if joint_prob > 0.9 else HAM



	
	
		






	




