import time
import os
import string
import itertools as itt
import numpy as np
import re
import collections
import pickle
from nltk.corpus import stopwords
import word2vec_read
import glove_read
import read_paragram


def processed_data_file_name(FLAGS):
  return \
    'processed_data-{}-d={}-rsw={}-mn={}-w={}-vsize={}.pkl'.format(
      FLAGS.embedding_type, FLAGS.embedding_size,
      FLAGS.remove_stopwords_from_s,
      FLAGS.map_numbers, FLAGS.win_sz, FLAGS.max_vocab_size)


# w2v_file = "/home/data/en_1000_no_stem/en.model"
# w2v_file = '/home/qv/nlp-data/enwiki-data/vocab_enwiki.txt'
# w2v_file = '/home/qv/nlp-data/enwiki-data/sentence_induced_emb.txt'

data_set_types = ['train', 'dev', 'test']
valid_chars = set(string.ascii_letters + string.digits + '_')
stopwords_set = set(stopwords.words('english'))


def select_embedding(FLAGS):
  if FLAGS.embedding_type == 'paragram':
    assert(FLAGS.embedding_size == 300)
    module = read_paragram
  elif FLAGS.embedding_type == 'glove':
    module = glove_read
  elif FLAGS.embedding_type == 'enwiki-skipgram':
    assert(FLAGS.embedding_size == 300)
    module = glove_read
  elif FLAGS.embedding_type == 'GoogleNews':
    assert(FLAGS.embedding_size == 300)
    module = word2vec_read

  return (FLAGS.w2v_file, module)


def read_embeddings(FLAGS, train_vocab, select_size):
  w2v_file, module = select_embedding(FLAGS)
  return module.read_embeddings(w2v_file, train_vocab, select_size)


def extract_ascii_words(FLAGS):
  w2v_file, module = select_embedding(FLAGS)
  return module.extract_ascii_words(w2v_file)


def timeit(orig_fn):
    def new_fn(*args, **kwargs):
        t = time.time()
        ret = orig_fn(*args, **kwargs)
        print('time = {} '.format(time.time() - t))
        return ret
    return new_fn


def is_number(st, FLAGS):
    if not FLAGS.map_numbers:
      return False
    try:
        float(st)
        return True
    except ValueError:
        return False


def reduceByKey(list_of_pairs, reduce_func=None):
    d = {}
    for a, b in list_of_pairs:
        if a in d:
            d[a].append(b)
        else:
            d[a] = [b]

    return {k: reduce_func(v) if reduce_func else v for k, v in d.items()}


@timeit
def build_vocab(w2v, reserved_vocab):
  vocab = reserved_vocab + w2v.keys()
  word2idx = {w: i for i, w in enumerate(vocab)}

  word_vec_size = len(w2v[w2v.keys()[0]])
  i2v = np.zeros((len(vocab), word_vec_size))
  for w, i in word2idx.items():
    if w in w2v:
      i2v[i] = w2v[w]

  return (word2idx, i2v)


def doc_sentence_id(st):
  doc_id, s_id = st.split('-')
  return int(doc_id[1:]), int(s_id)


def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stopwords_set]


def bigram_update(tokens, vocab):
  result = []
  i = 0
  while i < len(tokens):
      for n in range(10, 1, -1):
          w = '_'.join(tokens[i:i + n])
          if w in vocab:
              result.append(w)
              i = i + n
              break
      else:
          result.append(tokens[i])
          i = i + 1
  return result


def q_preprocess(q_str, FLAGS):
    if not FLAGS.map_numbers:
      return q_str
    return re.sub(r'\bhow many\b|\bHow many\b|\bHOW MANY', '_number_', q_str)


def tokenize(st, FLAGS):
  st = ''.join(
    [c if c in valid_chars else ' ' for c in st])
  return ['_number_' if is_number(t, FLAGS) else t for t in st.split()]


def remove_stopwords_from_q(tokens, FLAGS):
  if FLAGS.remove_stopwords_from_q:
    return remove_stopwords(tokens)
  else:
    return tokens


def remove_stopwords_from_s(tokens, FLAGS):
  if FLAGS.remove_stopwords_from_s:
    return remove_stopwords(tokens)
  else:
    return tokens


def parse_questions(data, w2v_vocab, FLAGS):
  qs = {int(qn[1:]): remove_stopwords_from_q(bigram_update(
        tokenize(q_preprocess(q, FLAGS), FLAGS), w2v_vocab), FLAGS)
        for qn, q, _, _, _, _, _ in data}
  return {qn: q for qn, q in qs.items() if len(q)}


def parse_sentences(data, w2v_vocab, FLAGS):
  return {doc_sentence_id(ds_id): remove_stopwords_from_s(bigram_update(
          tokenize(s, FLAGS), w2v_vocab), FLAGS)
          for _, _, _, _, ds_id, s, l in data}


def idfs(sentences):
    doc_freqs = collections.Counter([w for st in data_set_types
                                     for s in sentences[st].values()
                                     for w in set(s)])
    N = sum(len(sentences[st]) for st in data_set_types)
    return {w: np.log(1.0 * N / n) for w, n in doc_freqs.items()}


def matching_words_count(q, s, idfs):
    q = set([w.lower() for w in remove_stopwords(q)])
    s = set(remove_stopwords(s))
    count = sum(1 for w in s if w in q or w.lower() in q)
    idf_wtd_count = sum(idfs[w] for w in s if w in q or w.lower() in q)
    return (count, idf_wtd_count)


def q_to_sentence_map(data, questions, sentences, idfs):
  qs_pairs = [(int(q[1:]), (doc_sentence_id(ds_id), int(l)))
              for q, _, _, _, ds_id, _, l in data]
  
  qs_pairs = [(qn, (s_id, l) + matching_words_count(questions[qn],
                                                    sentences[s_id], idfs))
              for qn, (s_id, l) in qs_pairs
              if qn in questions and len(sentences[s_id])]
  q_to_s_map = reduceByKey(qs_pairs)
  return {q: list(np.random.permutation(slist))
          for q, slist in q_to_s_map.items()}


@timeit
def parse_corpus(FLAGS):

  data_file = {}
  data_file['train'] = os.path.join(FLAGS.input_dir, 'WikiQA-train.tsv')
  data_file['dev'] = os.path.join(FLAGS.input_dir, 'WikiQA-dev.tsv')
  data_file['test'] = os.path.join(FLAGS.input_dir, 'WikiQA-test.tsv')

  w2v_vocab = extract_ascii_words(FLAGS)
  w2v_vocab = {w: i for i, w in enumerate(w2v_vocab)}

  corpus = {st: [line.split('\t') for line in file(data_file[st])][1:]
            for st in data_set_types}
  questions = {st: parse_questions(corpus[st], w2v_vocab, FLAGS)
               for st in data_set_types}
  sentences = {st: parse_sentences(corpus[st], w2v_vocab, FLAGS)
               for st in data_set_types}
  q_to_s_map = {st: q_to_sentence_map(corpus[st], questions[st],
                                      sentences[st], idfs(sentences))
                for st in data_set_types}

  return (questions, sentences, q_to_s_map)


@timeit
def corpus_vocab(questions, sentences):
  corpus_words = list(itt.chain(*itt.chain(*[questions[st].values()
                                             for st in data_set_types])))
  corpus_words += list(itt.chain(*itt.chain(*[sentences[st].values()
                                              for st in data_set_types])))
  vocab = collections.Counter(corpus_words)
  vocab = {w: i for i, w in enumerate(vocab.keys())}
  return vocab


def vectorize_window(win, vocab, FLAGS):
  return [vocab[w] if w in vocab else vocab[FLAGS.unknown_word]
          for w in win]


def sentence_to_windows(s, win_sz, vocab, FLAGS):
  wins = []
  while len(s) > win_sz:
      wins.append(s[:win_sz])
      s = s[win_sz:]
  wins.append(s)
  return [vectorize_window(win, vocab, FLAGS) for win in wins]


@timeit
def vectorized_questions(questions, max_q_len, vocab, FLAGS):
    return {qn: [vocab[w] if w in vocab else vocab[FLAGS.unknown_word]
                 for w in q]
            for qn, q in questions.items()}


@timeit
def vectorized_sentences(sentences, max_s_len, vocab, FLAGS):
    return {s_id: [vocab[w] if w in vocab else vocab[FLAGS.unknown_word]
                   for w in s]
            for s_id, s in sentences.items()}


@timeit
def q_to_kv_pairs(q_to_s_map, sentences, win_sz, vocab, FLAGS):
  return {qn: [(win, s_id)
               for s_id, _, _, _ in s_list
               for win in sentence_to_windows(sentences[s_id],
                                              win_sz, vocab, FLAGS)]
          for qn, s_list in q_to_s_map.items()}


@timeit
def prepare_data(FLAGS):
  def step():
    i = 0
    while True:
      i += 1
      yield i

  processed_data_file = os.path.join(FLAGS.data_dir, FLAGS.data_pkl_file)
  if os.path.exists(processed_data_file):
    print 'Reading Processed data from file:{}'.format(processed_data_file)
    with open(processed_data_file, 'rb') as pkl_file:
      data_dict = pickle.load(pkl_file)
    return data_dict

  step = step()

  print('{} Parsing Corpus'.format(step.next()))
  questions, sentences, q_to_s_map = parse_corpus(FLAGS)
  max_q_len = max(max(len(q) for q in questions[st].values())
                  for st in data_set_types)
  max_s_len = max(max(len(s) for s in sentences[st].values())
                  for st in data_set_types)
  max_q_sents = max(max(len(slist) for slist in q_to_s_map[st].values())
                    for st in data_set_types)
  
  print('{}. Calculating Corpus Vocabulary'.format(step.next()))
  vocab = corpus_vocab(questions, sentences)
  
  print('{}. Reading Word2Vec Embeddings'.format(step.next()))
  t = time.time()
  w2v = read_embeddings(FLAGS, vocab,
                        max(len(vocab), FLAGS.max_vocab_size))
  print('time = {} '.format(time.time() - t))

  print('Corpus Vocabulary Size: {}'.format(len(vocab)))
  print('Vocabulary covered by pretrained Vectors: {}'.format(
    len(set(w2v.keys()) & set(vocab))))
  print('{}. Building Numpy Array for Embeddings'.format(step.next()))
  vocab, i2v = build_vocab(w2v, [FLAGS.pad_word, FLAGS.unknown_word,
                                 FLAGS.number_word])
  rev_vocab = {i: w for w, i in vocab.items()}

  print('{}. Vectorizing Questions'.format(step.next()))
  v_qs = {st: vectorized_questions(questions[st], max_q_len,
                                   vocab, FLAGS)
          for st in data_set_types}

  print('{}. Vectorizing Sentences'.format(step.next()))
  v_sentences = {st: vectorized_sentences(sentences[st], max_s_len,
                                          vocab, FLAGS)
                 for st in data_set_types}

  print('{}. Computing Question to Key-Value Pairs Mapping'.format(step.next()))
  q_to_kv_maps = {st: q_to_kv_pairs(q_to_s_map[st], sentences[st],
                                    FLAGS.win_sz, vocab, FLAGS)
                  for st in data_set_types}

  data_dict = {}
  data_dict['vocab'] = vocab
  data_dict['rev_vocab'] = rev_vocab
  data_dict['q_to_s_map'] = q_to_s_map
  data_dict['i2v'] = i2v
  data_dict['v_qs'] = v_qs
  data_dict['v_sentences'] = v_sentences
  data_dict['q_to_kv_maps'] = q_to_kv_maps
  data_dict['max_q_len'] = max_q_len
  data_dict['max_s_len'] = max_s_len
  data_dict['max_q_sents'] = max_q_sents

  with open(processed_data_file, 'wb') as pkl_file:
      pickle.dump(data_dict, pkl_file)
  return data_dict


if __name__ == "__main__":
  class FLAGS:
    pass

  FLAGS.win_sz = 7
  FLAGS.embedding_size = 300
  FLAGS.embedding_type = 'enwiki-skipgram'

  FLAGS.max_vocab_size = 100000
  FLAGS.pad_word = '_NIL'
  FLAGS.unknown_word = '_UNK'
  FLAGS.number_word = '_number_'

  FLAGS.map_numbers = False
  FLAGS.remove_stopwords_from_q = False
  FLAGS.remove_stopwords_from_s = True

  FLAGS.input_dir = '/home/qv/nlp-data/WikiQACorpus'
  FLAGS.data_dir = '/home/qv/wikiqa-data'
  FLAGS.data_pkl_file = processed_data_file_name(FLAGS)

  data_dict = prepare_data(FLAGS)
  vocab = data_dict['vocab']
  print "Vocabulary Size: ", len(vocab)
