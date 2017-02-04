import collections
import pprint, copy
import numpy as np
import random
import string


def read_data(data):
  with open(data, "r") as data_arr:
    for line in data_arr.readlines():
      yield line

class POS:
  """docstring for ."""
  def __init__(self):
    self.train = read_data("./hw3-data/train.txt")
    self.test_data = read_data("./hw3-data/test.txt")
    self.tags_bigrams_count = dict()
    self.markov = dict()
    self.vocab = collections.Counter()
    self.tags = collections.Counter()
    self.baseline()
    self.virterbi()
    self.virterbi_extra()


  def baseline(self):
    pp = pprint.PrettyPrinter(indent=4)
    print('---------Training--------')
    print('Choices include using the most common tag if the given tag is not in those that have been seen before')
    print('For the start and end symbol they have been added as a tag and sent to the tag bigram structure for counting')
    self.tags_bigrams_count['<s>'] = collections.Counter()
    self.tag_words_count = dict()
    # self.word_bigrams_tag_count = dict()
    for i, line in enumerate(self.train):
      line = line.rstrip().split(' ')
      last_tag = '<s>'
      tag = '<s>'
      last_word = '<s>'
      curr_word = '<s>'
      for word in line:
        if word == '':
          continue
        word = word.split('/')
        curr_word = word[0]
        tag = word[1]
        if tag not in self.tags_bigrams_count:
          self.tags_bigrams_count[tag] = collections.Counter( )
        # if word[1] not in self.word_bigrams_tag_count:
        #     self.word_bigrams_tag_count[word[1]] = dict()
        # if last_word not in self.word_bigrams_tag_count[word[1]]:
        #     self.word_bigrams_tag_count[word[1]][last_word] = collections.Counter()
        if tag not in self.tag_words_count:
          self.tag_words_count[ tag ] = collections.Counter( )
        if word[0] not in self.markov:
          self.markov[word[0]] = collections.Counter()
        self.tag_words_count[ tag ][ word[0]] += 1
        self.tags_bigrams_count[last_tag][tag] += 1
        self.markov[word[0]][tag] += 1
        self.vocab[word[0]] +=1
        # self.word_bigrams_tag_count[word[1]][last_word][curr_word] += 1
        if word[1] != '<s>' and word[1] != '</s>':
          self.tags[word[1]] += 1
        last_tag = tag
        last_word = curr_word
      self.tags_bigrams_count[last_tag]['</s>'] +=1
    self.tags_bigrams_prob = {}

    # tag probs bigrams
    vocSize = len(self.vocab)
    for tag in self.tags_bigrams_count:
      self.tags_bigrams_prob[tag] = {}
      sums_bigrams = sum(self.tags_bigrams_count[tag].values())
      things = set(self.tags_bigrams_count[tag])
      things.add('</s>')
      for tag_two in things:
        if tag_two in self.tags_bigrams_count[tag]:
          self.tags_bigrams_prob[tag][tag_two] = float(self.tags_bigrams_count[tag][tag_two]+0.001) / (sums_bigrams + vocSize)
        else:
          self.tags_bigrams_prob[tag][tag_two] = 0.001 / (sums_bigrams + vocSize)

    # word probs tag bigrams: part 3
    # self.word_bigrams_tag_prob = dict()
    # for tag in self.word_bigrams_tag_count:
    #   self.word_bigrams_tag_prob[tag] = {}
    #   for word_next in self.word_bigrams_tag_count[tag]:
    #       self.word_bigrams_tag_prob[tag][word_next] = dict()
    #       sums_bigrams = sum(self.word_bigrams_tag_count[tag][word_next].values())
    #       things = set(self.word_bigrams_tag_count[tag][word_next])
    #       things.add('</s>')
    #       for word_last in things:
    #         if word_last in self.word_bigrams_tag_count[tag][word_next]:
    #           self.word_bigrams_tag_prob[tag][word_next][word_last] = float(self.word_bigrams_tag_count[tag][word_next][word_last]+0.001) / (sums_bigrams + vocSize)
    #         else:
    #           self.word_bigrams_tag_prob[tag][word_next][word_last] = 0.001 / (sums_bigrams + vocSize)

    # tag P(w|t)
    self.tag_word_probs = dict()
    for tag in self.tag_words_count:
      self.tag_word_probs[tag] = dict()
      summed = float( sum(self.tag_words_count[tag].values() ) )
      for word in self.tag_words_count[tag]:
        self.tag_word_probs[tag][word] = float(self.tag_words_count[tag][word] + 0.001 )  / (summed + vocSize )

    for tag in self.tag_word_probs:
      print('Prob: (you|',tag,') = ',end='')
      if 'you' not in self.tag_word_probs[tag]:
        print('0')
        continue
      pp.pprint(self.tag_word_probs[tag]['you'])
    self.test()

  def test(self):
    print('---------Testing----------')
    correct = 0
    guess = 0
    i = 0
    line_output = ''
    test = self.test_data
    for line in test:
      if i == 1:
        line_output = line
      line = line.strip().split(' ')
      for word in line:
        if word == '':
          continue
        word = word.split('/')
        if word[0] not in self.markov:
          most = self.tags.most_common(1)[0]
        else:
          most = self.most_common(word[0])[0]
        if most == word[1]:
          correct += 1
        guess += 1
      i +=1
    line = line_output.rstrip().split(' ')
    print('---------Line 2--------')
    output = ''
    for word in line:
      if word == '':
        continue
      word = word.split('/')

      if word[0] not in self.markov:
        most = self.tags.most_common(1)[0]
      else:
        most = self.most_common(word[0])[0]
      if most == "N":
        output += ' ' + word[0]
    print(output)
    print('-----------------------')
    print("Accuracy for baseline is", correct/guess)

  def virterbi(self):
    pp = pprint.PrettyPrinter(indent=4)
    # viterbi[q0] ← 1
    # viterbi[q] ← 0 for q ̸= q0
    # for each state q′ in topological order do
    #   for each incoming transition q → q′ with weight p do
    #     if viterbi[q]× p > viterbi[q′] then
    #       viterbi[q′] ← viterbi[q]× p
    #       pointer[q′] ← q
    #     end if
    #   end for
    # end for
    pp.pprint(self.tags_bigrams_prob)
    print('---------Viterbi----------')
    tag_state = []
    pointer_state = []
    correct = 0
    guesses = 0
    test = read_data("./hw3-data/test.txt")
    output = ''
    for j, line in enumerate(test, 1):
      tag_state = []
      pointer_state = []
      tag_state.append({"<s>": 0})
      pointer_state.append({"<s>": 0})
      line = line.strip().split(' ')

      #make states to transition to
      for i in range(1, len(line)+1):
        tag_state.append(dict())
        pointer_state.append(dict())
        for tag in self.tags:
          tag_state[i][tag] = 0
          pointer_state[i][tag] = 0
      tag_state.append({"</s>": 0})
      pointer_state.append({"</s>": 0})

      # we need the languages states in a state machine
      # forward[q0] ← 1
      # forward[q] ← 0 for q ̸= q0
      # for each state q′ in topological order do
      #   for each incoming transition q → q′ with weight p do
      #     forward[q′] ← forward[q′]+forward[q]× p
      #   end for
      # end for
      states = self.ss_setup(len(line), self.tags, 0)
      for i, word in enumerate(line, 1):
        word = word.split('/')

        for to_state in states[i]:
          for from_state in states[i-1]:
            if word[0] in self.tag_word_probs[to_state]:
              prob = np.log(self.tags_bigrams_prob[from_state][to_state]*self.tag_word_probs[to_state][word[0]])
            else:
              prob = np.log(self.tags_bigrams_prob[from_state][to_state]*0.001/float(len(self.vocab)))
            cost_is = tag_state[i-1][from_state] + prob
            self.update_cost(cost_is, tag_state, pointer_state, i, to_state, from_state)

      for from_state in states[len(line)]:
        prob = np.log(self.tags_bigrams_prob[from_state]['</s>'])
        cost_is = tag_state[len(line)][from_state] + prob
        self.update_cost(cost_is, tag_state, pointer_state, len(line)+1, '</s>', from_state)

      steps = self.ss_setup(len(line), 0, 0)
      i = len(line)
      steps[i] = '</s>'
      while i > 0:
        steps[i-1] = pointer_state[i+1][steps[i]]
        i -= 1

      for i, word in enumerate(line):
        if i == 1:
          output
        word = word.split('/')
        guessed_tag = steps[i]
        if j == 2:
          if guessed_tag == 'N':
            output += word[0] +' '
        if guessed_tag == word[1]:
          correct += 1
        guesses += 1
    print("----------Line 2 for viterbi----------")
    print(output)
    print("Accuracy for viterbi is", correct/guesses)

  def virterbi_extra(self):
    pp = pprint.PrettyPrinter(indent=4)
    print('---------Viterbi: Part 3----------')
    tag_state = []
    pointer_state = []
    correct = 0
    guesses = 0
    test = read_data("./hw3-data/test.txt")
    output = ''
    for j, line in enumerate(test, 1):
      tag_state = []
      pointer_state = []
      tag_state.append({"<s>": 0})
      pointer_state.append({"<s>": 0})
      line = line.strip().split(' ')

      #make states to transition to
      for i in range(1, len(line)+1):
        tag_state.append(dict())
        pointer_state.append(dict())
        for tag in self.tags:
          tag_state[i][tag] = 0
          pointer_state[i][tag] = 0
      tag_state.append({"</s>": 0})
      pointer_state.append({"</s>": 0})

      states = self.ss_setup(len(line), self.tags, 0)
      last_word = '<s>'
      curr_word = '<s>'
      for i, word in enumerate(line, 1):
        word = word.split('/')
        curr_word = word[0]
        for to_state in states[i]:
          for from_state in states[i-1]:
            if word[0] in string.punctuation:
              if from_state == to_state:
                  prob = np.log(1)
              else:
                  prob = np.log(0.001)
            elif word[0] in self.tag_word_probs[to_state]:
              prob = np.log(self.tags_bigrams_prob[from_state][to_state]*self.tag_word_probs[to_state][word[0]])
            else:
              prob = np.log(self.tags_bigrams_prob[from_state][to_state]*0.001/float(len(self.vocab)))
            cost_is = tag_state[i-1][from_state] + prob
            self.update_cost(cost_is, tag_state, pointer_state, i, to_state, from_state)
        last_word = curr_word

      for from_state in states[len(line)]:
        prob = np.log(self.tags_bigrams_prob[from_state]['</s>'])
        cost_is = tag_state[len(line)][from_state] + prob
        self.update_cost(cost_is, tag_state, pointer_state, len(line)+1, '</s>', from_state)

      steps = self.ss_setup(len(line), 0, 0)
      i = len(line)
      steps[i] = '</s>'
      while i > 0:
        steps[i-1] = pointer_state[i+1][steps[i]]
        i -= 1

      for i, word in enumerate(line):
        if i == 1:
          output
        word = word.split('/')
        guessed_tag = steps[i]
        if j == 2:
          if guessed_tag == 'N':
            output += word[0] +' '
        if guessed_tag == word[1]:
          correct += 1
        else:
          print("Wrong", word[0],guessed_tag, word[1])
        guesses += 1
    print("----------Line 2 for viterbi part3----------")
    print(output)
    print("Accuracy for viterbi part3 is", correct/guesses)

  def ss_setup(self, word_count, type_app, range_start):
    states = []
    states.append(["<s>"])
    for i in range(range_start, word_count+1):
      states.append(type_app)
    states.append(["</s>"])
    return states

  def update_cost(self,cost_is, tag_state, pointer_state, i_val, to_state, from_state):
    if cost_is > tag_state[i_val][to_state] or tag_state[i_val][to_state] is 0:
      tag_state[i_val][to_state] = cost_is
      pointer_state[i_val][to_state] = from_state

  def most_common(self, word):
    if self.markov[word].most_common(1)[0] == '<s>':
      return self.markov[word].most_common(1)[1]
    return self.markov[word].most_common(1)[0]
