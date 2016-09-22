from test import hw1_nb_test, hw1_logistic_test, hw1_nb_bigrams_test, hw1_logistic_bigrams_test, hw1_nb_trigrams_test, hw1_logistic_trigrams_test
from train import train
from train_bigrams import train as train_bigrams
from train_trigrams import train as train_trigrams
import pickle


def main():
    print('Overall Implementation choice, at times I used pickling to run the ')
    print('data into the structure once and then loaded from the pickle file in')
    print('order to save time by not rerunning documents I had already read over.')
    print('this saved a lot of time in the testing phase. \n\n')
    candidates, total_documents, unique_words, wordProbs = train('hw1-data/train')
    # candidates, total_documents, unique_words, wordProbs = pickle.load( open( "save.p", "rb" ))
    hw1_nb_test(candidates, total_documents, unique_words, wordProbs)

    hw1_logistic_test(candidates, total_documents, unique_words, wordProbs)

    print('\n\nRunning the bigrams on the documents. The accuracy seems to go down, and ')
    print('the bigrams run alone without the single words to add in do not have much ')
    print('accuracy to them. Only yielding .075. I assume that bigrams normally work')
    print('much better but because of my specific implementation there is trouble with')
    print('these specifically. The logistic bigrams hovered around the same. I assume ')
    print('there is some part of my code that is keeping them from getting over this hump.')

    candidates_bigrams, total_documents_bigrams, unique_words_bigrams, wordProbs_bigrams = train_bigrams('hw1-data/train')
    # candidates_bigrams, total_documents_bigrams, unique_words_bigrams, wordProbs_bigrams = pickle.load( open( "save_bigram.p", "rb" ))
    hw1_nb_bigrams_test(candidates_bigrams, total_documents_bigrams, unique_words_bigrams, wordProbs_bigrams )

    hw1_logistic_bigrams_test(candidates_bigrams, total_documents_bigrams, unique_words_bigrams, wordProbs_bigrams)

    print('\n\nGoing with trigrams for my third to see if the accuracy further declines.\n\n')

    candidates_trigrams, total_documents_trigrams, unique_words_trigrams, wordProbs_trigrams = train_trigrams('hw1-data/train')
    # candidates_trigrams, total_documents_trigrams, unique_words_trigrams, wordProbs_trigrams = pickle.load( open( "save_trigram.p", "rb" ))
    hw1_nb_trigrams_test(candidates_trigrams, total_documents_trigrams, unique_words_trigrams, wordProbs_trigrams )

    hw1_logistic_trigrams_test(candidates_trigrams, total_documents_trigrams, unique_words_trigrams, wordProbs_trigrams)

if __name__ == "__main__":
    main()
