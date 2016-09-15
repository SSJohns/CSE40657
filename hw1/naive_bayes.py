import numpy as np
import scipy


class Naive_Bayes:
	def __init__(self, candidates, unique_words, wordProbs, total_documents):
		self.wordProbs = wordProbs
		self.candidates = candidates
		self.unique_words = unique_words
		self.total_documents = total_documents

		self.c_k = dict()
		self.p_k = dict()
		self.sum_c_k_w = dict()
		self.p_k = dict()
		self.p_k_w = dict()

		for whom in self.candidates:
			self.p_k_w[whom] = dict()

		self.func_c_k()
		self.sum_c_k = self.func_sum_c_k()
		self.func_calc_p_k()


	def func_c_k(self):
		for whom in self.candidates:
			self.c_k[whom] = self.candidates[whom][0]

	def func_sum_c_k(self):
		sum = 0
		for value in self.c_k.values():
			sum = sum + value
		return sum

	def func_c_k_w(self, whom, word):
		if word not in self.candidates[whom][2]:
			return 0
		return self.candidates[whom][2][word]

	def func_sum_c_k_w(self, word):
		sum = 0
		if word in self.sum_c_k_w:
			return self.sum_c_k_w[word]
		for whom in self.candidates:
			sum = sum + self.func_c_k_w(whom, word)
		self.sum_c_k_w[word] = sum
		return sum

	def func_calc_p_k(self):
		summed = 0
		for whom in self.candidates:
			self.p_k[whom] = self.candidates[whom][0]/float(self.sum_c_k)
			summed = summed + self.p_k[whom]
		self.sum_p_k = summed

	def func_calc_p_k_w(self, whom, word):
		if word in self.p_k_w[whom]:
			return self.p_k_w[whom][word]
		self.p_k_w[whom][word] = (self.func_c_k_w(whom, word) + 0.02)/float(self.func_sum_c_k_w(word) + len(self.unique_words)*0.02)
		return self.p_k_w[whom][word]

	def func_sum_log_p_w_k(self, doc):
		p_logs = dict()
		sum_logs = 0
		for word in doc:
			for candidate in self.candidates:
				sum_logs = sum_logs + self.func_calc_p_k_w(candidate, word)
		return sum_logs

	def func_sum__e_logs_p_k_plus_sum_log_p_w_k(self, word, doc, total):
		sum_k_primes = []
		addition = self.func_sum_log_p_w_k(doc)
		for k_prime in self.candidates:
			sum_k_primes.append(np.log(self.p_k[k_prime]) + addition)

		log_sum_k_primes = scipy.misc.logsumexp(sum_k_primes)
		return log_sum_k_primes

	def naive_bayes(self, candidates, total, doc, unique_words):
		temp_p_k_d = dict()
		probablity_speaking = dict()
		z_d = 0

		# p_k
		sum_p_k = self.sum_p_k

		for whom in candidates:
			# Sum p_w_k in document
			sum_log_p_k_w = 0.0
			for word in doc:
				sum_log_p_k_w = sum_log_p_k_w + np.log(self.func_calc_p_k_w(whom, word))

			# print("Summing found: ", np.log(sum_p_k), sum_log_p_k_w, self.func_sum__e_logs_p_k_plus_sum_log_p_w_k(word, doc, total))
			sum_logs = np.log(sum_p_k) + sum_log_p_k_w # - self.func_sum__e_logs_p_k_plus_sum_log_p_w_k(word, doc, total)
			# sum_logs = np.exp(sum_logs)
			temp_p_k_d.update({whom: sum_logs})

		for whom in self.candidates:
			probablity_speaking[whom] = np.exp(temp_p_k_d[whom])

		for whom in temp_p_k_d:
			p_k_d = scipy.misc.logsumexp(temp_p_k_d[whom])
			temp_p_k_d.update({whom: p_k_d})

		max = np.finfo('d').min
		winner = []
		for whom in temp_p_k_d:
			p_k_d = temp_p_k_d[whom]
			# print("For candidate: ", whom, " naive_bayes for doc is ", p_k_d)
			if p_k_d > max:
				max = p_k_d
				winner = [whom, p_k_d]
		return winner, probablity_speaking
