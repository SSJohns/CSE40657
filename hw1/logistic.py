import autograd.numpy as np
from autograd.core import primitive
from autograd import grad
import scipy.misc
import autograd

from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle, random


class Logistic:
	def __init__(self, classes, unique_words):
		self.logsumexp = primitive(scipy.misc.logsumexp)
		self.lamb_doll = dict()
		self.lamb_doll_word = dict()
		for category in classes:
			self.lamb_doll[category] = 0
			self.lamb_doll_word[category] = dict()
			self.lamb_doll_word[category][""] = 0.0
			for word, count in classes[category][2].items():
				self.lamb_doll_word[category][word] = 0.0
		self.categories = classes
		with open("hw1-data/dev") as dev:
		    train_content = dev.readlines()
		    self.dev_lines= [self.spoken_word(line) for line in train_content]

	def lambda_k_plus_sum_lambdas(self):
		pass

	def spoken_word(self, line):
		# word tokenizer, removes punctuation
		tokenizer = RegexpTokenizer(r'\w+')

		# word lemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()

		# stopword eliminator
		stop = set(stopwords.words('english'))

		line = line.split(' ', 1)
		line_split = tokenizer.tokenize(line[1])
		line_fill = []

		for word in line_split:
			if word not in stop:
				line_fill.append( wordnet_lemmatizer.lemmatize(word) )
		return (line[0], line_fill)

	def count(self, category, word):
		if word in self.categories[category]:
			return self.categories[category][word]
		else:
			return 0

	def delta(self, k, k_i):
		if k is k_i:
			return 1
		else:
			return 0

	def p_k_given_d(self, doc, category):
		p_k_given_d_speakers = dict()
		sum_lambda_k_w = 0
		sum_logsumexp = []
		for cat in self.categories:
			sum_lambda_k_w = self.lamb_doll_word[cat][""]
			for word in doc:
				if word in self.lamb_doll_word[cat]:
					sum_lambda_k_w += self.lamb_doll_word[cat][word]
				else:
					self.lamb_doll_word[cat][word] = 0
			p_k_given_d_speakers[cat] = sum_lambda_k_w
			sum_logsumexp.append(sum_lambda_k_w)

		return self.lamb_doll_word[category][""] + sum_lambda_k_w - self.logsumexp(sum_logsumexp)

	def p_k_given_d_speakers(self, doc):
		# print(doc)
		p_k_d = dict()
		sum_p_k_w = 0
		for cat in self.categories:
			sum_p_k_w = self.lamb_doll_word[cat][""]
			for word in doc:
				if word in self.lamb_doll_word[cat]:
					# print(cat, word, self.lamb_doll_word[cat][word])
					sum_p_k_w += self.lamb_doll_word[cat][word]
				else:
					self.lamb_doll_word[cat][word] = 0.01
			p_k_d[cat] = sum_p_k_w
		sum_val = 0
		for key, val in p_k_d.items():
			# print(key, val)
			p_k_d[key] = np.exp(val)
			sum_val += np.exp(val)
		return {cat: val/sum_val for cat, val in p_k_d.items()}


	def guess_speaker(self, doc):
		all_value = self.p_k_given_d_speakers(doc)
		spoken_by = max(all_value, key=lambda x: all_value[x])
		# print('Spoken_by',spoken_by, all_value[spoken_by])
		return spoken_by

	def test_dev_accuracy(self):

		random.shuffle(self.dev_lines)
		guesses = total_guessed = 0
		for doc in self.dev_lines:
			speaker_guess = self.guess_speaker(doc[1])
			# print(speaker_guess, doc[0])
			if speaker_guess == doc[0]:
				guesses = guesses + 1
			total_guessed = total_guessed + 1
		print('Guessed', guesses, 'out of', total_guessed, 'docs')

	def train(self, training_file):
		# classes = {}
		file = open(training_file, 'r')

		# word tokenizer, removes punctuation
		tokenizer = RegexpTokenizer(r'\w+')

		# word lemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()

		# stopword eliminator
		stop = set(stopwords.words('english'))

		eta = 0.01
		for i in range(1,10):
			neg_log_prob = 0
			print("Iteration: ", i)
			for value in self.lamb_doll_word:
				self.lamb_doll_word[value] = dict()
				self.lamb_doll_word[value][""] = 0.0
			for line in file:	#d_i
				line = line.split(' ', 1)
				line_split = tokenizer.tokenize(line[1])
				line_fill = []

				for word in line_split:
					if word not in stop:
						line_fill.append( wordnet_lemmatizer.lemmatize(word) )

				# training with zeroed lambdas
				for category in self.categories:
					p_k_given_d = self.p_k_given_d_speakers(line_fill)
					neg_log_prob += p_k_given_d[category]
					for word in line_fill + [""]:
						if word in self.lamb_doll_word[category]:
							self.lamb_doll_word[category][word] = self.lamb_doll_word[category][word] + self.count(category, word)
						else:
							self.lamb_doll_word[category][word] = self.count(category,word)
						for category_i in self.categories:
							if word in self.lamb_doll_word[category_i]:
								self.lamb_doll_word[category_i][word] -= eta*p_k_given_d[category_i]*self.count(category_i, word)
							if category_i is category:
								self.lamb_doll_word[category][""] += 0.01
			print("Negative log prob", -1*np.log(neg_log_prob))
			self.test_dev_accuracy()

		print("--------------------------------------------------------------")
		print("--------------------------------------------------------------")
		print("--------------------2_b lambda values-------------------------")
		print("--------------------------------------------------------------")
		print("--------------------------------------------------------------")

		print('lambda(clinton), lambda(clinton,president), lambda(clinton,country)', self.lamb_doll_word['clinton'][''], self.lamb_doll_word['clinton']['country'], self.lamb_doll_word['clinton']['president'])
		print('lambda(trump), lambda(trump,president), lambda(trump,country)', self.lamb_doll_word['trump'][''], self.lamb_doll_word['trump']['country'], self.lamb_doll_word['trump']['president'])

		pickle.dump([self.lamb_doll, self.lamb_doll_word], open("logistic.p", "wb"))

		file.close()

	def logistic(self):
		pass
