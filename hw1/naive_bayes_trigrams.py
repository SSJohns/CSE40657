import numpy as np
import scipy


class Naive_Bayes_Trigrams:
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
		self.sum_c_k = 0
		self.func_c_k()
		self.func_calc_p_k()


	def func_c_k(self):
		for whom in self.candidates:
			self.c_k[whom] = self.candidates[whom][0]
			self.sum_c_k += self.candidates[whom][0]

	def func_c_k_w(self, whom, word):
		if word not in self.candidates[whom][2]:
			return 0
		return self.candidates[whom][2][word]

	def func_calc_p_k(self):
		summed = 0
		print('p(k):')
		for whom in self.candidates:
			self.p_k[whom] = self.candidates[whom][0]/float(self.total_documents)
			print('p(k)',whom,':',self.p_k[whom])
			summed = summed + self.p_k[whom]
		self.sum_p_k = summed

	def func_p_k_given_d(self, whom, doc):
		sum_doc = np.log(self.p_k[whom])
		for word in doc:
			sum_doc += np.log(self.func_calc_p_k_w(whom,word))
		return sum_doc

	def func_speak_given_state(self, doc):
		""" p(k|d)"""
		values = {}
		for speaker in self.candidates:
			check = self.func_p_k_given_d(speaker, doc)
			values.update({speaker: check})
		mav_value = max(values.values())
		prob_values = {}
		for val in values:
			prob_values.update({val:values[val] + mav_value})
		sum_vals = sum(prob_values.values())
		values = {}
		for speak, valss in prob_values.items():
			values.update({speak:valss/sum_vals})
		return values

	def func_calc_p_k_w(self, whom, word):
		if word in self.p_k_w[whom]:
			return self.p_k_w[whom][word]
		self.p_k_w[whom][word] = (self.func_c_k_w(whom, word) + 0.02)/float(self.candidates[whom][1] + self.unique_words*0.02)
		return self.p_k_w[whom][word]

	def func_sum_log_p_w_k(self, doc):
		sum_logs = 0
		for word in doc:
			for candidate in self.candidates:
				sum_logs = sum_logs + np.log(self.func_calc_p_k_w(candidate, word))
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

		# p_k
		sum_p_k = self.sum_p_k

		for whom in candidates:
			# Sum p_w_k in document
			sum_log_p_k_w = 0.0
			for word in doc:
				sum_log_p_k_w = sum_log_p_k_w + np.log(self.func_calc_p_k_w(whom, word))

			if len(doc) > 2:
				firstWord = doc[0]
				secondWord = doc[1]
				for word in doc[2:]:
					trigram = firstWord + ' ' + secondWord + ' '+ word
					sum_log_p_k_w = sum_log_p_k_w + np.log(self.func_calc_p_k_w(whom, trigram))
					firstWord = secondWord
					secondWord = word

			sum_logs = np.log(sum_p_k) + sum_log_p_k_w
			temp_p_k_d.update({whom: sum_logs})
		winner = max(temp_p_k_d, key=temp_p_k_d.get)

		return winner, probablity_speaking
