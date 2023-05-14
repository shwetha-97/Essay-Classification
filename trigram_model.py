import sys
from collections import defaultdict
import math
import random
import os
import os.path

def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    new_sequence = []
    num_starts = 1
    start = 0
    if n >= 3:
        num_starts = n - 1
    new_sequence.extend(["START"] * num_starts)
    new_sequence.extend(sequence)
    new_sequence.append("STOP")

    ngram_list = []
    while start <= len(new_sequence) - n:
        ngram = new_sequence[start: start + n]
        ngram_list.append(tuple(ngram))
        start += 1
    return ngram_list


class TrigramModel(object):

    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.total_word_count = 0

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)

        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        for sentence in corpus:
            unigram_list = get_ngrams(sentence, 1)
            bigram_list = get_ngrams(sentence, 2)
            trigram_list = get_ngrams(sentence, 3)

            for unigram in unigram_list:
                if unigram != ('START',):
                    self.unigramcounts[unigram] += 1

            for bigram in bigram_list:
                self.bigramcounts[bigram] += 1

            for trigram in trigram_list:
                self.trigramcounts[trigram] += 1

            self.total_word_count += len(unigram_list) - 1 # Excluding (START,) for every sentence
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[(trigram[0], trigram[1])] == 0:
            return self.unigramcounts[(trigram[2],)] / self.total_word_count
        else:
            return self.trigramcounts[trigram] / self.bigramcounts[(trigram[0], trigram[1])]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[(bigram[0],)] == 0:
            return self.unigramcounts[(bigram[1],)] / self.total_word_count
        else:
            return self.bigramcounts[bigram] / self.unigramcounts[(bigram[0],)]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        # return count of unigram/total number of tokens -> excluding START but including STOP (using sum of all values
        # in the unigram count dict)
        if unigram == ('START',):
            return 0
        return self.unigramcounts[unigram] / self.total_word_count

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        count = 0
        current_w = None
        current_bigram_context = ('START', 'START')
        lst = ['']
        result = []
        while count < t or current_w != 'STOP' and len(lst) != 0:
            # get a list of words starting with (current_bigram_context) - using trigram keys
            lst = self.get_trigrams_with_context(current_bigram_context)
            # get a random word from the list (set current_w) and add it to result
            current_w = random.choice(lst)
            # update result
            result.append(current_w)
            # update current_bigram_context to (current_bigram_context[1], current_w)
            current_bigram_context = (current_bigram_context[1], current_w)
            count += 1
        return result

    def get_trigrams_with_context(self, context):
        lst = []
        not_allowed = ['UNK', '**', '#', '&', '-rrb-', '-lrb-']
        for trigram in self.trigramcounts.keys():
            if trigram[0] == context[0] and trigram[1] == context[1] and trigram[2] not in not_allowed:
                lst.append(trigram[2])
        return lst

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability((trigram[1], trigram[2])) + lambda3 * self.raw_unigram_probability((trigram[2],))

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        logprob = 0.0
        for trigram in trigrams:
            if self.smoothed_trigram_probability(trigram) != 0:
                logprob += math.log2(self.smoothed_trigram_probability(trigram))
        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        logprobs = 0.0
        test_word_count = 0
        for sentence in corpus:
            test_word_count += len(sentence) + 1
            logprobs += self.sentence_logprob(sentence)
        l = logprobs / test_word_count
        return math.pow(2, -l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    # class = high
    for f in os.listdir(testdir1):
        pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp_high < pp_low :
            correct += 1
        total += 1

    # class = low
    for f in os.listdir(testdir2):
        pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        if pp_low < pp_high:
            correct += 1
        total += 1

    return float(correct/total) * 100


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])
    print(model.generate_sentence())
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    # print(acc)
