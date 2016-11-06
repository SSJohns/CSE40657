import collections
from itertools import islice
import numpy as np


def window(seq, n=2):
	"Returns a sliding window (of width n) over data from the iterable"
	"   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
	it = iter(seq)
	result = tuple(islice(it, n))
	if len(result) == n:
		yield result
	for elem in it:
		result = result[1:] + (elem,)
		yield result

def window_next(seq, n=2):
	table = window(seq,n)
	t_rand = list()
	feats = list()
	feats.extend(table)
	for i in range(len(feats)-1):
		t_rand.append( ( feats[i], feats[i+1][len(feats[i+1])-1] ) )
	return t_rand

class Language_Model:
	def __init__(self,language):
		print('--------- ENGLISH ---------')
		if language == 'english':
			self.train = self.read_data('hw2-files/english/train') # ('toy') #
			self.dev = self.read_data('hw2-files/english/dev')
			self.test = self.read_data('hw2-files/english/test')
		if language == 'chinese':
			self.train = self.read_data('hw2-files/chinese/train.han')
			self.dev = self.read_data('hw2-files/chinese/dev.han')
			self.test = self.read_data('hw2-files/chinese/test.han')
		### Collect counts
		self.unigram = set()
		self.c_u = collections.Counter()
		self.c_uw = collections.defaultdict(lambda: set())
		self.c_u_dot = collections.defaultdict( lambda: collections.defaultdict(lambda: 0.0000000000000000000001) )
		self.lambda_value = dict()
		self.p_w = dict()
		self.count = 0
		self.ppl = 0
		self.ppl_n = 0
		self.training()

	### Read in the data

	def read_data(self, fn):
		for line in open(fn):
			yield line

	### Features
	def gen_features(self,words, n_grams):
		feats = []  # always-on feature
		feats.extend(words) # unigram features
		self.count += len(words)
		# grabs all indivdual words
		for a in words:
			self.unigram.update(a)

		# collects n-gram chars from 2 to n
		for i in range(2,n_grams+1):
			feats.extend(window(words, i))

		# adds these to our models
		for feat in feats:
			if len(feat) <= 1:
				self.c_u_dot[len(feat)-1][''] +=1
			else:
				self.c_u_dot[len(feat)-1][ feat[0:len(feat)-1] ] +=1
		self.c_u.update(feats)
		self.vocab.update(feats)

	# return nr1+ for the lambda calculation
	def n_r(self, u):
		sum = 0
		for a in self.unigram:
			new_val = u+(a,)
			if new_val in self.c_u_dot[len(new_val)]:
				sum += 1
		return sum


	def lambda_u(self,u):
		# memoization
		if u in self.lambda_value:
			return self.lambda_value[u]
		self.lambda_value[u] = self.c_u_dot[len(u)][u]/float( self.c_u_dot[len(u)][u] + self.n_r(u) )
		if self.lambda_value[u] <= 0.00000001:
			import ipdb; ipdb.set_trace()
		return self.lambda_value[u]

	# recursively find our prob values
	def phrase_helper(self, phrase):
		if len(phrase) == 1:
			div = self.count
			lambda_val = self.c_u_dot[len(phrase)-1]['']/float( self.c_u_dot[len(phrase)-1][''] + self.n_r(phrase[-1:]) )
			return lambda_val*(self.c_u[phrase[0]]/float( div )) + (1-lambda_val)*1/(len(self.vocab)+1)
		else:
			self.p_w[phrase] = self.lambda_u(phrase[-1:])*(self.c_u[phrase]/float( self.c_u_dot[len(phrase)-1][phrase[:-1]] )) + ( (1.-self.lambda_u(phrase[-1:]))*self.phrase_helper(phrase[1:]) )
			return self.p_w[phrase]

	# grab all unique chars and
	# check if they are the next value
	def phrase_find(self, phrase):
		probs = dict()
		for u in self.unigram:
			probs.update({u:self.phrase_helper(phrase+(u,))})
		summed = probs[max(probs, key=probs.get)]
		if summed != 1:
			self.ppl += np.log(summed)
		self.ppl_n += 1
		return max(probs, key=probs.get)

	def test_dev(self, gram_model):
		print('---------- DEV ----------')
		i = 0
		for phrase in self.dev:
			checks = list()
			for j in range(0, gram_model):
				phrase = '🐟'+phrase
			checks.extend(window_next(phrase, gram_model-1)) # collision
			for ch in checks:
				a = self.phrase_find(ch[0])
				print(a, "guessed, answer is:", ch[1])
				i += 1
				if i >= 10:
					return

	# check our training model
	def test_phrases(self, gram_model):
		print('---------- TEST ----------')
		correct = 0
		guesses = 0
		for phrase in self.test:
			checks = list()
			for j in range(2, gram_model+1):
				phrase = '🐟'+phrase
			checks.extend(window_next(phrase, gram_model-1)) # collision
			for ch in checks:
				a = self.phrase_find(ch[0])
				if a == ch[1]:
					correct += 1
				guesses += 1
				# print(correct/guesses)
		print("For Test Accuracy is", correct/float(guesses) )

	def training(self):
		### Collect counts
		gram_model = 6

		self.vocab = set()
		i = 0

		# grab our fetures and
		# train on the input file
		for u in self.train:
			for j in range(2, gram_model+1):
				u = '🐟'+u
			self.gen_features(u,gram_model)
			if (i % 10000) == 0:
				print(i/100000 * 100, "%")
			i+=1

		self.vocab.add('<unk>')

		self.test_dev(gram_model)
		### find lambda's of phrases
		self.test_phrases(gram_model)

		ppl = self.ppl/(self.ppl_n+1)
		# print(ppl)
		print('Perplexity:',np.exp (-1*ppl ))
