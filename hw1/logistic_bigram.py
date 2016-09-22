import autograd.numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle, random


class Logistic_Bigrams:
	def __init__(self, classes, unique_words):
		self.lamb_doll_word = dict()
		for category in classes:
			self.lamb_doll_word[category] = dict()
			for word, count in classes[category][2].items():
				self.lamb_doll_word[category][word] = 0.0
		self.categories = classes
		with open("hw1-data/dev") as dev:
		    train_content = dev.readlines()
		    self.dev_lines= [self.spoken_word(line) for line in train_content]
		with open("hw1-data/test") as test:
			train_content = test.readlines()
			self.test_lines= [self.spoken_word(line) for line in train_content]

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

	def p_k_given_d_speakers(self, doc):
		p_k_d = dict()
		sum_p_k_w = 0
		for cat in self.categories:
			sum_p_k_w = self.lamb_doll_word[cat][""]
			for word in doc:
				if word in self.lamb_doll_word[cat]:
					sum_p_k_w += self.lamb_doll_word[cat][word]
			p_k_d[cat] = sum_p_k_w
		sum_val = .1
		for key, val in p_k_d.items():
			p_k_d[key] = np.exp(val)
			sum_val += np.exp(val)

		return {cat: val/sum_val for cat, val in p_k_d.items()}


	def guess_speaker(self, doc):
		all_value = self.p_k_given_d_speakers(doc)
		spoken_by = max(all_value, key=lambda x: all_value[x])
		return spoken_by

	def test_test_accuracy(self):

		# random.shuffle(self.test_lines)
		guesses = total_guessed = 0
		for doc in self.test_lines:
			speaker_guess = self.guess_speaker(doc[1])
			if speaker_guess == doc[0]:
				guesses = guesses + 1
			total_guessed = total_guessed + 1
		print('Guessed', guesses, 'out of', total_guessed, 'docs. Accuracy: ', guesses/total_guessed, "\n")

	def test_dev_accuracy(self):

		# random.shuffle(self.test_lines)
		guesses = total_guessed = 0
		for doc in self.dev_lines:
			speaker_guess = self.guess_speaker(doc[1])
			if speaker_guess == doc[0]:
				guesses = guesses + 1
			total_guessed = total_guessed + 1
		print('Guessed', guesses, 'out of', total_guessed, 'docs. Accuracy: ', guesses/total_guessed,"\n")

	def train(self):
		# classes = {}
		file_t = open('hw1-data/train', 'r')
		self.train_file = file_t.readlines()
		file_t.close()

		eta = 0.1
		for i in range(1,16):
			neg_log_prob = 0
			eta = eta * 0.95
			random.shuffle(self.train_file)
			print("Iteration: ", i)
			for value in self.lamb_doll_word:
				self.lamb_doll_word[value] = dict()
				self.lamb_doll_word[value][""] = 0.0
			for category in self.categories:
				self.lamb_doll_word[category] = dict()
				self.lamb_doll_word[category][""] = 0.0
				for word, count in self.categories[category][2].items():
					self.lamb_doll_word[category][word] = eta
			for line in self.train_file:	#d_i
				line_fill = self.spoken_word(line)

				# training with zeroed lambdas
				category = line_fill[0]
				p_k_given_d = self.p_k_given_d_speakers(line_fill[1])
				for word in (line_fill[1] + [""]):
					if word in self.lamb_doll_word[category]:
						self.lamb_doll_word[category][word] = self.lamb_doll_word[category][word] + eta
					for category_i in self.categories:
						if word in self.lamb_doll_word[category_i]:
							self.lamb_doll_word[category_i][word] -= eta*p_k_given_d[category_i]

				# bigram
				if len(line_fill[1]) > 1:
					new_doc = []
					prevWord = line_fill[1][0]
					for word in line_fill[1][1:]:
						bigram = prevWord + ' ' + word
						new_doc.append(bigram)
						prevWord = word
					p_k_given_d_bigram = self.p_k_given_d_speakers(new_doc)
					for val in p_k_given_d:
						p_k_given_d[val] += p_k_given_d_bigram[val]
					for word in (new_doc + [""]):
						if word in self.lamb_doll_word[category]:
							self.lamb_doll_word[category][word] = self.lamb_doll_word[category][word] + eta
						else:
							self.lamb_doll_word[category][word] = eta
						for category_i in self.categories:
							if word in self.lamb_doll_word[category_i]:
								self.lamb_doll_word[category_i][word] -= eta*p_k_given_d[category_i]

				neg_log_prob -= np.log(p_k_given_d[category])
			print("Negative log prob", neg_log_prob)
			self.test_dev_accuracy()
			# input('Waiting...')
		print("--------------------------------------------------------------")
		print("--------------------------------------------------------------")
		print("--------------------2_b lambda values-------------------------")
		print("--------------------------------------------------------------")
		print("--------------------------------------------------------------")

		print('lambda(clinton), lambda(clinton,president), lambda(clinton,country)', self.lamb_doll_word['clinton'][''], self.lamb_doll_word['clinton']['country'], self.lamb_doll_word['clinton']['president'])
		print('lambda(trump), lambda(trump,president), lambda(trump,country)', self.lamb_doll_word['trump'][''], self.lamb_doll_word['trump']['country'], self.lamb_doll_word['trump']['president'])

		pickle.dump(self.lamb_doll_word, open("logistic.p", "wb"))

		# self.test_test_accuracy()
