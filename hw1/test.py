from naive_bayes import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def hw1_nb_test(candidates, total_documents, unique_words, wordProbs):
	nb = Naive_Bayes(candidates, unique_words, wordProbs, total_documents)

	print('Clinton c(k): ', nb.c_k['clinton'], ' c(clinton, country) ',
		nb.func_c_k_w('clinton','country'),
		' c(clinton,president) ', nb.func_c_k_w('clinton','president'))
	print('Trump c(k):', nb.c_k['trump'], ' c(trump, country) ',
		nb.func_c_k_w('trump','country'), ' c(trump,president) ',
		nb.func_c_k_w('trump','president'))

	print('Clinton p(clinton)', calc_p_k(candidates,'clinton',total_documents),
		' p(clinton,president) ', calc_p_k_w(candidates,'clinton','president', unique_words),
		' p(clinton,country) ', calc_p_k_w(candidates,'clinton','country', unique_words))
	print('Trump p(trump)', calc_p_k(candidates,'trump',total_documents))


	# word tokenizer, removes punctuation
	tokenizer = RegexpTokenizer(r'\w+')

	# word lemmatizer
	wordnet_lemmatizer = WordNetLemmatizer()

	# stopword eliminator
	stop = set(stopwords.words('english'))

	## Naive-Bayes equation
	value_correct = value_total = 0
	with open('./hw1-data/test','r') as f:
		for dev_file in f:
			line_fill = []
			dev_file = dev_file.split(' ', 1)
			for word in dev_file[1].split(' '):
				if word not in stop:
					line_fill.append( wordnet_lemmatizer.lemmatize(word) )
			winner, prob_speak = nb.naive_bayes(candidates, total_documents, line_fill, unique_words)
			for whom in prob_speak:
				print(whom, prob_speak[whom])
			if winner[0] == dev_file[0]:
				value_correct = value_correct + 1
			value_total = value_total + 1
			print("Winner candidate: ", winner[0], "Answer: ", dev_file[0], " naive_bayes for doc is ", winner[1])
		print('Accuracy is ', value_correct/float(value_total))

def hw1_logistic_test():
    pass
