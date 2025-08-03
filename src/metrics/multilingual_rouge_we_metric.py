import collections
import gzip
from collections import Counter
from multiprocessing import Pool

import gin
import numpy as np
import six
# We use this implementation (https://pypi.org/project/snowballstemmer/) to allow for multilingual stemming
import snowballstemmer
from nltk.tokenize import RegexpTokenizer
from scipy import spatial
from summ_eval.metric import Metric


##############################################################################
# PREPROCESSING
##############################################################################

def normalize_word(word):
    return word.lower()

def get_all_content_words_stem(sentences, stem, tokenize=False, language="basque"):
    stemmer = snowballstemmer.stemmer(language)
    tokenizer = RegexpTokenizer(r'\w+')
    #
    all_words = []
    if tokenize:
        for s in sentences:
            if stem:
                all_words.extend([stemmer.stemWord(r) for r in tokenizer.tokenize(s)])
            else:
                all_words.extend(tokenizer.tokenize(s))
    else:
        if isinstance(sentences, list):
            all_words = sentences[0].split()
        else:
            all_words = sentences.split()
    #
    normalized_content_words = list(map(normalize_word, all_words))
    return normalized_content_words

def pre_process_summary_stem(summary, stem=True, tokenize=True, language="basque"):
    summary_ngrams = get_all_content_words_stem(summary, stem, tokenize=tokenize, language=language)
    return summary_ngrams


def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _has_embedding(ngram, embs):
    for w in ngram:
        if not w in embs:
            return False
    return True

def _get_embedding(ngram, embs):
    res = []
    for w in ngram:
        res.append(embs[w])
    return np.sum(np.array(res), 0)

def _find_closest(ngram, counter, embs):
    ## If there is nothing to match, nothing is matched
    if len(counter) == 0:
        return "", 0, 0

    ## If we do not have embedding for it, we try lexical matching
    if not _has_embedding(ngram, embs):
        if ngram in counter:
            return ngram, counter[ngram], 1
        else:
            return "", 0, 0

    ranking_list = []
    ngram_emb = _get_embedding(ngram, embs)
    for k, v in six.iteritems(counter):
        ## First check if there is an exact match
        if k == ngram:
            ranking_list.append((k, v, 1.))
            continue

        ## if no exact match and no embeddings: no match
        if not _has_embedding(k, embs):
            ranking_list.append((k, v, 0.))
            continue

        ## soft matching based on embeddings similarity
        k_emb = _get_embedding(k, embs)
        ranking_list.append((k, v, 1 - spatial.distance.cosine(k_emb, ngram_emb)))

    ## Sort ranking list according to sim
    ranked_list = sorted(ranking_list, key=lambda tup: tup[2], reverse=True)

    ## extract top item
    return ranked_list[0]

def _soft_overlap(peer_counter, model_counter, embs):
    THRESHOLD = 0.8
    result = 0
    for k, v in six.iteritems(peer_counter):
        closest, count, sim = _find_closest(k, model_counter, embs)
        if sim < THRESHOLD:
            continue
        if count <= v:
            del model_counter[closest]
            result += count
        else:
            model_counter[closest] -= v
            result += v

    return result

def rouge_n_we(peer, models, embs, n, alpha=0.5, return_all=False, tokenize=False, language="basque"):
    """
    Compute the ROUGE-N-WE score of a peer with respect to one or more models, for
    a given value of `n`.
    """

    if len(models) == 1 and isinstance(models[0], str):
        models = [models]
    peer = pre_process_summary_stem(peer, False, tokenize, language)
    models = [pre_process_summary_stem(model, False, tokenize, language) for model in models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _soft_overlap(peer_counter, model_counter, embs)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha, return_all)


def _convert_to_numpy(vector):
    return np.array([float(x) for x in vector])

def load_embeddings(filepath, vocab=None):
    dict_embedding = {}
    if vocab is not None:
        with gzip.open(filepath) as f:
            for line in f:
                line = line.decode().rstrip().split(" ")
                key = line[0]
                if key in vocab:
                    vector = line[1::]
                    dict_embedding[key.lower()] = _convert_to_numpy(vector)
    else:
        with gzip.open(filepath) as f:
            for line in f:
                line = line.decode().rstrip().split(" ")
                key = line[0]
                vector = line[1::]
                dict_embedding[key.lower()] = _convert_to_numpy(vector)
    return dict_embedding

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha, return_all=False):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        if return_all:
            return precision_score, recall_score, (precision_score * recall_score) / denom
        else:
            return (precision_score * recall_score) / denom
    else:
        if return_all:
            return precision_score, recall_score, 0.0
        else:
            return 0.0

##############################################################################
# METRIC
##############################################################################

@gin.configurable
class MultilingualRougeWeMetric(Metric):
    def __init__(self, emb_path, n_gram=3, \
                 n_workers=24, tokenize=True, lang="basque", vocab=None):
        """
        Multilingual ROUGE-WE metric
        Adapted from from https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b

        Args:
                :param emb_path: path to dependency-based word embeddings found here:
                        BASQUE: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz
                        SPANISH: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz
                :param n_gram: n_gram length to be used for calculation; if n_gram=3,
                        only calculates ROUGE-WE for n=3; reset n_gram to calculate
                        for other n-gram lengths
                :param n_workers: number of processes to use if using multiprocessing
                :param tokenize: whether to apply stemming and basic tokenization to input;
                        otherwise assumes that user has done any necessary tokenization

        """
        self.word_embeddings = load_embeddings(emb_path, vocab=vocab)
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(summary, list):
            summary = [summary]
        score = rouge_n_we(summary, reference, self.word_embeddings, self.n_gram, \
                 return_all=True, tokenize=self.tokenize)
        score_dict = {f"multilingual_rouge_we_{self.n_gram}_p": score[0], f"multilingual_rouge_we_{self.n_gram}_r": score[1], \
                      f"multilingual_rouge_we_{self.n_gram}_f": score[2]}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, references))
        p.close()
        if aggregate:
            corpus_score_dict = Counter()
            for x in results:
                corpus_score_dict.update(x)
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(summaries))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return True
