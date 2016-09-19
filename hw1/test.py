from naive_bayes import *
from logistic import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def hw1_nb_test(candidates, total_documents, unique_words, wordProbs):
	nb = Naive_Bayes(candidates, unique_words, wordProbs, total_documents)

	# word tokenizer, removes punctuation
	tokenizer = RegexpTokenizer(r'\w+')

	# word lemmatizer
	wordnet_lemmatizer = WordNetLemmatizer()

	# stopword eliminator
	stop = set(stopwords.words('english'))
	print("--------------------------------------------------------------")
	print("--------------------------------------------------------------")
	print("--------------------NAIVE-BAYES TEST--------------------------")
	print("--------------------------------------------------------------")
	print("--------------------------------------------------------------")

	print('Clinton c(k): ', nb.c_k['clinton'], ' \nc(clinton, country) ',
		nb.func_c_k_w('clinton','country'),
		' \nc(clinton,president) ', nb.func_c_k_w('clinton','president'))
	print('Trump c(k):', nb.c_k['trump'], ' \nc(trump, country) ',
		nb.func_c_k_w('trump','country'), ' \nc(trump,president) ',
		nb.func_c_k_w('trump','president'))

	print('Clinton p(clinton)', nb.p_k['clinton'],
		' \np(clinton,president) ', nb.func_calc_p_k_w('clinton','president'),
		' \np(clinton,country) ', nb.func_calc_p_k_w('clinton','country'))
	print('Trump p(trump)', nb.p_k['trump'],
		' \np(trump,president) ', nb.func_calc_p_k_w('trump','president'),
		' \np(trump,country) ', nb.func_calc_p_k_w('trump','country'))

	with open('./hw1-data/dev','r') as f:
		dev_line = f.readline()
		line_fill = []
		dev_file = dev_line.split(' ', 1)
		line_new = tokenizer.tokenize(dev_file[1])
		for word in line_new:
			if word not in stop:
				line_fill.append( wordnet_lemmatizer.lemmatize(word) )
		print('\n\np(k|d) dev_file line 1')
		spear_probs = nb.func_speak_given_state(line_fill)
		for val in spear_probs:
			print("P(k|d) of", val, "is", spear_probs[val])
		# input('Wainting on you')

	## Naive-Bayes equation
	value_correct = value_total = 0
	with open('./hw1-data/test','r') as f:
		for dev_file in f:
			line_fill = []
			dev_file = dev_file.split(' ', 1)
			line_new = tokenizer.tokenize(dev_file[1])
			for word in line_new:
				if word not in stop:
					line_fill.append( wordnet_lemmatizer.lemmatize(word) )
			winner, prob_speak = nb.naive_bayes(candidates, total_documents, line_fill, unique_words)
			# print(winner, dev_file[0])
			if winner == dev_file[0]:
				value_correct = value_correct + 1
			value_total = value_total + 1
			# print("Winner candidate: ", winner[0], "Answer: ", dev_file[0], " naive_bayes for doc is ", winner[1])
		print('Accuracy is ', value_correct/float(value_total))

def hw1_logistic_test(candidates, total_documents, unique_words, wordProbs):
	print("--------------------------------------------------------------")
	print("--------------------------------------------------------------")
	print("--------------------Logistic Regression-----------------------")
	print("--------------------------------------------------------------")
	print("--------------------------------------------------------------")
	lg = Logistic(candidates, unique_words)
	lg.train('hw1-data/train')
