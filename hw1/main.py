from test import hw1_nb_test
from train import train
import pickle
import sys


def main():
    # candidates, total_documents, unique_words, wordProbs = train(sys.argv[1])
    candidates, total_documents, unique_words, wordProbs = pickle.load( open( "save.p", "rb" ))
    hw1_nb_test(candidates, total_documents, unique_words, wordProbs)

if __name__ == "__main__":
	main()
