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

def keywithmaxval(d):
	""" a) create a list of the dict's keys and values;
		b) return the key with the max value"""
	v=list(d.values())
	k=list(d.keys())
	return k[v.index(max(v))]

def window_next(seq, n=2):
	table = window(seq,n)
	t_rand = dict()
	feats = list()
	feats.extend(table)
	# print(feats)
	for i in range(len(feats)-1):
		t_rand.update({ feats[i]: feats[i+1][len(feats[i+1])-1] })
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
		self.c_u = collections.defaultdict(lambda: 0.01)
		self.c_uw = collections.defaultdict(lambda: set())
		self.c_u_dot = collections.defaultdict( lambda: collections.defaultdict(lambda: 0.01) )
		self.lambda_value = dict()
		self.p_w = dict()

		self.training()

	### Read in the data

	def read_data(self, fn):
		for line in open(fn):
			yield line

	### Features
	def gen_features(self,words, n_grams):
		feats = ['<bias>']  # always-on feature
		feats.extend(words) # unigram features

		# grabs all indivdual words
		for a in words:
			self.unigram.update(a)

		# collects n-gram chars from 2 to n
		for i in range(2,n_grams+1):
			feats.extend(window(words, i))

		# adds these to our models
		for feat in feats:
			# import ipdb; ipdb.set_trace()
			self.c_uw[len(feat)].update(feat)
			if len(feat) <= 1:
				self.c_u_dot[len(feat)][feat] +=1
			else:
				self.c_u_dot[len(feat)][ feat[0:len(feat)] ] +=1
			self.c_u[feat] += 1
		self.vocab.update(feats)

	# return nr1+ for the lambda calculation
	def n_r(self, u):
		sum = 0
		for a in self.unigram:
			# import ipdb; ipdb.set_trace()
			new_val = u+(a,)
			if new_val in self.c_u_dot[len(new_val)]:
				sum += self.c_u_dot[len(new_val)][new_val]
		# return len(self.c_u_dot[len(new_val)].values())
		# print(sum, u, self.unigram)
		return sum


	def lambda_u(self,u):
		# print(u)
		count = 0

		# memoization
		if u in self.lambda_value:
			return self.lambda_value[u]
		# import ipdb; ipdb.set_trace()
		# print(u, self.c_u[u], len(self.c_u_dot[len(u)]))
		# nxt= set()
		for n in self.c_uw[len(u)+1]:
			# import ipdb; ipdb.set_trace()
			# print(n[-1:], u)
			if n[-1:] == u[-1:]:
				count += 1
				nxt.update(n[-1])
		if count != 0:
			self.lambda_value[u] = count / (count + len(nxt)+1)
		else:
			# c_u_dot is a dict of n lengths with a list of words
			# in those lengths and how many times weve seen them before
			self.lambda_value[u] = self.c_u_dot[len(u)][u]/( self.c_u_dot[len(u)][u] + self.n_r(u) ) # [len(u)][u[-1:]]
		return self.lambda_value[u]

	# recursively find our prob values
	def phrase_helper(self, phrase):
		# print(phrase)
		if phrase[0] == '🐟':
			if len(phrase) == 1:
				return 0
			return self.phrase_helper(phrase[1:])
		elif len(phrase) == 1:
			# print(self.c_u[phrase[0]], float(self.c_u_dot[phrase[0]]), (1-self.lambda_u(phrase) ) )
			return self.lambda_u(phrase)*(self.c_u[phrase]/float(self.c_u_dot[1][phrase])) + (1-self.lambda_u(phrase))
		else:
			# print('-----------------------------')
			# print(phrase)
			# print(phrase[0])
			# print('cu',self.c_u[phrase])
			# print('u*',self.c_u_dot[phrase[0]])
			# print('lamb: ', self.lambda_u(phrase))
			# print('lamb: ', self.lambda_u(phrase[0]))
			# print('lamb minus:',(1-self.lambda_u(phrase)))
			self.p_w[phrase] = self.lambda_u(phrase)*(self.c_u[phrase]/float( self.c_u_dot[len(phrase)][phrase[-1:]] )) + ( (1-self.lambda_u(phrase))*self.phrase_helper(phrase[1:]) )
			# print(self.p_w[phrase])
			return self.p_w[phrase]

	# grab all unique chars and
	# check if they are the next value
	def phrase_find(self, phrase):
		probs = dict()
		# print(self.c_u_dot)
		for u in self.unigram:
			# print(u)
			probs.update({u:self.phrase_helper(phrase+(u,))})
		# print(sum(probs.values()))
		return keywithmaxval(probs)

	# training
	def train_phrases(self, gram_model):
		for phrase in self.vocab:
			checks = dict()
			for j in range(2, gram_model):
				phrase = '🐟'+phrase
			checks.update(window_next(phrase, gram_model-1))
			for ch in checks:
				self.phrase_find(ch)

	# check our training model
	def test_phrases(self, gram_model):
		correct = 0
		guesses = 0
		# random.shuffle(self.dev)
		for phrase in self.dev:
			checks = dict()
			for j in range(2, gram_model):
				phrase = '🐟'+phrase
			checks.update(window_next(phrase, gram_model-1))
			for ch in checks:
				a = self.phrase_find(ch)
				print(a, "guessed, answer is:", checks[ch])
				# import ipdb; ipdb.set_trace()
				if a == checks[ch]:
					correct += 1
				guesses += 1
				print(correct/guesses)
		print("For Dev Accuracy is", correct/float(guesses) )

	def training(self):
		### Collect counts
		gram_model = 10

		self.vocab = set()
		i = 0

		# grab our fetures and
		# train on the input file
		for u in self.train:
			# for j in range(2, gram_model+1):
			# u = '🐟'+u
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

		### find lambda's of phrases

		self.test_phrases(gram_model)

		print(sum(self.p_w.values()))

	def prob_char(self,prev_char, curr_char):
		pass
