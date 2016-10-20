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
		# print(result)
		yield result

def window_next(seq, n=2):
	table = window(seq,n)
	t_rand = list()
	feats = list()
	feats.extend(table)
	# print(feats)
	for i in range(len(feats)-1):
		t_rand.append( ( feats[i], feats[i+1][len(feats[i+1])-1] ) )
	# t_rand.update( { feats[len(feats) -1]:'\n' })
	# print(t_rand)
	return t_rand

class Language_Model:
	def __init__(self,language):
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
			# import ipdb; ipdb.set_trace()
			# self.c_uw[len(feat)].update(feat)
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
			# import ipdb; ipdb.set_trace()
			new_val = u+(a,)
			# print('nr',new_val)
			# print(new_val in self.c_u_dot[len(new_val)])
			if new_val in self.c_u_dot[len(new_val)]:
				sum += 1
		# return len(self.c_u_dot[len(new_val)].values())
		# print(sum, u)
		return sum


	def lambda_u(self,u):
		# print(u)
		count = 0

		# memoization
		if u in self.lambda_value:
			return self.lambda_value[u]
		# import ipdb; ipdb.set_trace()
		# print(u, self.c_u[u], len(self.c_u_dot[len(u)]))
		# c_u_dot is a dict of n lengths with a list of words
		# in those lengths and how many times weve seen them before
		# print('c u*', self.c_u_dot[len(u)][u],self.c_u_dot[len(u)][u])
		# print('nr', self.n_r(u))
		self.lambda_value[u] = self.c_u_dot[len(u)][u]/float( self.c_u_dot[len(u)][u] + self.n_r(u) ) # [len(u)][u[-1:]]
		if self.lambda_value[u] <= 0.00000001:
			import ipdb; ipdb.set_trace()
		return self.lambda_value[u]

	# recursively find our prob values
	def phrase_helper(self, phrase):
		# print(phrase)
		# if phrase[0] == '🐟':
		# 	if len(phrase) == 1:
		# 		return 0
		# 	return (1-self.lambda_u(phrase[:-1]))*self.phrase_helper(phrase[1:])
		if len(phrase) == 1:
			# print(self.c_u[phrase[0]], float(self.c_u_dot[phrase[0]]), (1-self.lambda_u(phrase) ) )
			# print('-------------- SINGLE---------------')
			# print(phrase)
			# print(phrase)
			# print('lamb minus:',(1-self.lambda_u(phrase)))
			div = self.count # self.c_u_dot[1][phrase[0]])
			lambda_val = self.c_u_dot[len(phrase)-1]['']/float( self.c_u_dot[len(phrase)-1][''] + self.n_r(phrase[-1:]) )
			# print('lamb: ', lambda_val)
			return lambda_val*(self.c_u[phrase[0]]/float( div )) + (1-lambda_val)*1/(len(self.vocab)+1)
		else:
			# print('--------- DOUBLE --------------------')
			# print(phrase)
			# print(phrase[:-1])
			# print('cu',self.c_u[phrase])
			# print('lamb: ', self.lambda_u(phrase))
			# print('lamb minus:',(1-self.lambda_u(phrase)))
			self.p_w[phrase] = self.lambda_u(phrase[-1:])*(self.c_u[phrase]/float( self.c_u_dot[len(phrase)-1][phrase[-1:]] )) + ( (1.-self.lambda_u(phrase[-1:]))*self.phrase_helper(phrase[1:]) )
			# print(self.p_w[phrase])
			return self.p_w[phrase]

	# grab all unique chars and
	# check if they are the next value
	def phrase_find(self, phrase):
		probs = dict()
		# if phrase[-1:] in self.un:
		# 	for u in self.un[phrase[-1:]]:
		# 		probs.update({u:self.phrase_helper(phrase+(u,))})
		# else:
		for u in self.unigram:
			probs.update({u:self.phrase_helper(phrase+(u,))})
		return max(probs, key=probs.get)

	# training
	# def train_phrases(self, gram_model):
	# 	for phrase in self.vocab:
	# 		checks = dict()
	# 		for j in range(2, gram_model):
	# 			phrase = '🐟'+phrase
	# 		checks.update(window_next(phrase, gram_model-1))
	# 		for ch in checks:
	# 			self.phrase_find(ch)

	# check our training model
	def test_phrases(self, gram_model):
		correct = 0
		guesses = 0
		# random.shuffle(self.dev)
		for phrase in self.test:
			checks = list()
			for j in range(2, gram_model+1):
				phrase = '🐟'+phrase
			checks.extend(window_next(phrase, gram_model-1)) # collision
			for ch in checks:
				a = self.phrase_find(ch[0])
				# print(a, "guessed, answer is:", ch[1])
				# import ipdb; ipdb.set_trace()
				if a == ch[1]:
					correct += 1
				guesses += 1
				# print(correct/guesses)
		print("For Test Accuracy is", correct/float(guesses) )

	def training(self):
		### Collect counts
		gram_model = 7

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
		# print("CU:", self.c_u)
		# print("Dot", self.c_u_dot)
		# import ipdb; ipdb.set_trace()
		self.c_u_dot[1]['🐟'] = 1
		self.c_u['🐟'] = 1
		# import ipdb; ipdb.set_trace()
		self.vocab.add('<unk>')

		self.un = dict()
		for u in self.c_u_dot[2]:
			# import ipdb; ipdb.set_trace()
			if u[0] in self.un:
				self.un[ u[0] ].update(u[1])
			else:
				self.un[ u[0] ] = set()
				self.un[ u[0] ].update(u[1])
		### find lambda's of phrases
		# import ipdb; ipdb.set_trace()
		self.test_phrases(gram_model)

		print(sum(self.p_w.values()))

	def prob_char(self,prev_char, curr_char):
		pass
