import autograd.numpy as np
from autograd import grad

from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle

class Logistic:
	def __init__(self, model, doc, classes):
		self.lamb_doll = dict()
		self.lamb_doll_word = dict()
		for category in classes:
			self.lamb_doll_word[category] = dict()
		self.catogories = classes

	def lambda_k_plus_sum_lambdas(self):
		pass

	def count(self, category):
		if word in self.categories[category]:
			return self.categories[category][word]
		else:
			return 0.01

	def delta(self, k, k_i):
		if k is k_i:
			return 1
		else:
			return 0

	def log_p_k_given_d(self, doc, category):
		sum_lambda_k_w = 0
		sum_logsumexp = []
		for cat in self.categories:
			sum_lambda_k_w_plus = 0
			for word in doc:
				if word in self.lamb_doll_word[category]:
					sum_lambda_k_w += self.lamb_doll_word[cat][word]
				else:
					self.lamb_doll_word[category][word] = 0
			sum_lambda_k_w_plus += self.lamb_doll[cat] + sum_lambda_k_w
			sum_logsumexp.append(sum_lambda_k_w_plus)
		p_k_d = []
		for category in self.categories:
			p_k_d.append(self.lamb_doll[category] + sum_lambda_k_w - autograd.scipy.misc.logsumexp(sum_logsumexp))
			count(doc,category)*(self.delta())
		return L_i

	def train(self, training_file):
		classes = {}
		file = open(training_file, 'r')
		total_documents = 0
		unique_words = Counter()
		wordProbs = dict()

		# word tokenizer, removes punctuation
		tokenizer = RegexpTokenizer(r'\w+')

		# word lemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()

		# stopword eliminator
		stop = set(stopwords.words('english'))

		eta = 1/8
		for iteration in range(1:10):
			for line in file:	#d_i
				line = line.split(' ', 1)
				line_split = tokenizer.tokenize(line[1])
				line_fill = []

				for word in line_split:
					if word not in stop:
						line_fill.append( wordnet_lemmatizer.lemmatize(word) )

				# training with zeroed lambdas
				L_i = self.log_p_k_given_d(doc, category)
				for lamdba in self.lamb_doll:
					self.lamb_doll[lambda] += eta * L_i


		file.close()
		pickle.dump([candidates,total_documents,unique_words,wordProbs], open( "save.p", "wb" ))
		return candidates, total_documents, unique_words, wordProbs

	def logistic(self):
		pass
