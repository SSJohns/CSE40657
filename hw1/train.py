from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle

## Main data structure
## { candidate : [ #docs, #words, Counter()] }
def train(training_file):
	candidates = {}
	file = open(training_file, 'r')
	total_documents = 0
	unique_words = Counter()
	wordProbs = dict()

	# word tokenizer, removes punctuation
	tokenizer = RegexpTokenizer(r'\w+')

	# word lemmatizer
	wordnet_lemmatizer = WordNetLemmatizer()

	# stopword eliminator
	stop = set(stopwords.words('english'))

	for line in file:
		line = line.split(' ', 1)
		line_split = tokenizer.tokenize(line[1])
		line_fill = []

		for word in line_split:
			if word not in stop:
				line_fill.append( wordnet_lemmatizer.lemmatize(word) )

		# for word in line_split:
		# 	line_fill.append(word)

		if line[0] in candidates:
			candidates[line[0]][0] = candidates[line[0]][0] + 1
			past_counter = candidates[line[0]][2]
			total_documents = total_documents + 1
			candidates[line[0]][2] = Counter(line_fill) + past_counter
			unique_words = unique_words + Counter(line_fill)
			candidates[line[0]][1] = candidates[line[0]][1] + len(line_fill)
		else:
			total_documents = total_documents + 1
			unique_words = unique_words + Counter(line_fill)
			candidates.update({line[0]:[1, len(line_fill), Counter(line_fill)]})
			wordProbs[line[0]] = dict()

	for candidate in candidates:
		candidates[candidate][2] = candidates[candidate][2] + Counter({'unk':0})
		value = 0
		for word in candidates[candidate][2]:
			wordProbs[candidate][word] = candidates[candidate][2][word] / len(candidates[candidate][2])

	file.close()
	pickle.dump([candidates,total_documents,unique_words,wordProbs], open( "save.p", "wb" ))
	return candidates, total_documents, unique_words, wordProbs
