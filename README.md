# Essay Classification with n-gram Language Models
## Extracting n-grams
`get_ngrams` - Takes a list of strings and an integer n as input, and returns padded n-grams over the list of strings. The result should be a list of Python tuples. 

## Counting n-grams
`count_ngrams` - Counts the occurrence frequencies for ngrams in the corpus. The method already creates three instance variables of TrigramModel, which store the unigram, bigram, and trigram counts in the corpus. Each variable is a dictionary (a hash map) that maps the n-gram to its count in the corpus. 

## Raw unsmoothed probabilities
`raw_trigram_probability`, `raw_bigram_probability`, `raw_unigram_probability` - Each of these methods returns an unsmoothed probability computed from the trigram, bigram, and unigram counts.

## Smoothed probabilities
`smoothed_trigram_probability` - Uses linear interpolation between the raw trigram, unigram, and bigram probabilities, seeing the interpolation parameters to lambda1 = lambda2 = lambda3 = 1/3

## Compute sentence probability
`sentence_logprob` - Returns the log probability of an entire sequence. We use the get_ngrams function to compute trigrams and the smoothed_trigram_probability method to obtain probabilities. We convert each probability into logspace using math.log2

## Computing perplexity of model
`perplexity` - Compute the perplexity of the model on an entire corpus, which we can test using brown_test.txt

## Essay classification
We apply the trigram model to a text classification task. We use a data set of essays written by non-native speakers of English for the ETS TOEFL test. These essays are scored according to skill level low, medium, or high. We only consider essays that have been scored as "high" or "low". We train a different language model on a training set of each category and then use these models to automatically score unseen essays. We compute the perplexity of each language model on each essay. The model with the lower perplexity determines the class of the essay. 

## Generate Sentence
`generate_sentence` - return a list of strings, randomly generated from the raw trigram model. Keep track of the previous two tokens in the sequence, starting with ("START","START"). Then, to create the next word, we look at all words that appeared in this context and get the raw trigram probability for each.
