import collections
from itertools import islice


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

class Language_Model2:
	def __init__(self,language):
		print('--------- CHINESE ---------')
		self.train = self.read_data('hw2-files/chinese/train.han')
		self.cmap = self.read_data('hw2-files/chinese/charmap')
		self.devpin = self.read_data('hw2-files/chinese/dev.pin')
		self.devhan = self.read_data('hw2-files/chinese/dev.han')
		self.read_data('hw2-files/chinese/test.pin')
		self.testpin = self.read_data('hw2-files/chinese/test.pin')
		self.testhan = self.read_data('hw2-files/chinese/test.han')
		### Collect counts
		self.unigram = set()
		self.c_u = collections.Counter()
		self.c_uw = collections.defaultdict(lambda: set())
		self.c_u_dot = collections.defaultdict( lambda: collections.defaultdict(lambda: 0.01) )
		self.lambda_value = dict()
		self.p_w = dict()
		self.count = 0

		self.chrmap = collections.defaultdict(lambda: set())
		for line in self.cmap:
			line = line.rstrip()
			line = line.split(' ')
			self.chrmap[line[1]].update(line[0])
		# import ipdb; ipdb.set_trace()
		print('Possibilities of yi:', len(self.chrmap['yi']))
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
		self.lambda_value[u] = self.c_u_dot[len(u)][u]/float( self.c_u_dot[len(u)][u] + self.n_r(u) ) # [len(u)][u[-1:]]
		return self.lambda_value[u]

	# recursively find our prob values
	def phrase_helper(self, phrase):
		if len(phrase) == 1:
			div = self.count # self.c_u_dot[1][phrase[0]])
			lambda_val = self.c_u_dot[len(phrase)-1]['']/float( self.c_u_dot[len(phrase)-1][''] + self.n_r(phrase) )
			return lambda_val*(self.c_u[phrase[0]]/float( div )) + (1-lambda_val)*1/(len(self.vocab)+1)
		else:
			self.p_w[phrase] = self.lambda_u(phrase[:-1])*(self.c_u[phrase]/float( self.c_u_dot[len(phrase)-1][phrase[:-1]] )) + ( (1.-self.lambda_u(phrase[:-1]))*self.phrase_helper(phrase[1:]) )
			return self.p_w[phrase]

	# grab all unique chars and
	# check if they are the next value
	def phrase_find(self, phrase, chin_phra):
		probs = dict()
		possibles = list()
		# import ipdb; ipdb.set_trace()
		if phrase[-1:][0] in self.chrmap:
			if len(phrase[-1:][0]) == 1:
				possibles.extend(phrase[-1:][0])
			possibles.extend(self.chrmap[phrase[-1:][0]])
		else:
			return phrase[-1:][0]
		for u in possibles:
			# import ipdb; ipdb.set_trace()
			probs.update({u:self.phrase_helper(chin_phra[0]+(u,))})
		# import ipdb; ipdb.set_trace()
		return max(probs, key=probs.get)

	def test_dev(self, gram_model):
		print('---------- DEV ----------')
		i = 0
		# import ipdb; ipdb.set_trace()
		for phrase, ans in zip(self.devpin, self.devhan):
			ans = ans.rstrip()
			phrase = phrase.rstrip()
			ans_char = list()
			for char in ans:
				ans_char.append(char)
			phrase = phrase.split(' ')
			checks = list()
			ans_checks = list()
			for j in range(0, gram_model-1):
				phrase = ['🐟'] + phrase
				ans_char = ['🐟'] + ans_char
			checks.extend(window_next(phrase, gram_model-1)) # collision
			ans_checks.extend( window_next(ans_char, gram_model-1) )
			for ch, ah in zip(checks, ans_checks):
				# print(ch)
				# import ipdb; ipdb.set_trace()
				a = self.phrase_find(ch, ah)
				print(a, "guessed, answer is:", ah[-1:][0])
				i += 1
				if i >= 10:
					return

	# check our training model
	def test_phrases(self, gram_model):
		correct = 0
		guesses = 0
		print('------- TEST ---------')
		# random.shuffle(self.dev)
		for phrase, ans in zip(self.testpin, self.testhan):
			ans = ans.rstrip()
			phrase = phrase.rstrip()
			phrase = phrase.split(' ')
			ans_char = list()
			for char in ans:
				ans_char.append(char)
			checks = list()
			ans_checks = list()
			for j in range(0, gram_model-1):
				phrase = ['🐟'] + phrase
				ans_char = ['🐟'] + ans_char
			checks.extend(window_next(phrase, gram_model-1)) # collision
			ans_checks.extend( window_next(ans_char, gram_model-1) )
			for ch, ah in zip(checks, ans_checks):
				a = self.phrase_find(ch, ah)
				if a == '<space>':
					a = ' '
				# print(a, "guessed, answer is:", ah[-1:][0])
				if a == ah[-1:][0]:
					# print('Correct')
					correct += 1
				guesses += 1
		print("For Test Accuracy is", correct/float(guesses) )

	def training(self):
		### Collect counts
		gram_model = 2

		self.vocab = set()
		i = 0

		# grab our fetures and
		# train on the input file
		for u in self.train:
			for j in range(0, gram_model-1):
				u = '🐟'+u
			self.gen_features(u,gram_model)
			if (i % 10000) == 0:
				print(i/100000 * 100, "%")
			i+=1
		self.vocab.add('<unk>')
		self.test_dev(gram_model)
		### find lambda's of phrases
		self.test_phrases(gram_model)

		# print(sum(self.p_w.values()))
