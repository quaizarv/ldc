import gensim
import time
import string

valid_chars = string.ascii_letters + '_' + '.' + string.digits


def timeit(orig_fn):
  def new_fn(*args, **kwargs):
    t = time.time()
    ret = orig_fn(*args, **kwargs)
    print time.time() - t
    return ret
  return new_fn


def extract_ascii_words(emb_file):
  if '.bin' in emb_file:
    model = gensim.models.KeyedVectors.load_word2vec_format(
      emb_file, binary=True)
  else:
    model = gensim.models.Word2Vec.load(emb_file)
  return [k for k in model.vocab.keys()
          if all(c in valid_chars for c in k)]


def read_embeddings(emb_file, train_vocab, select_size):
  '''Extract embeddings from the embedding file corresponding to the passed
    train vocabulary 'train_vocab' and some more.  If |train_vocab| >
    |select_size| then pick |train_vocab| - |select_size| most frequent
    elements (in the corpus on which embeddings were trained) as well.
  '''
  if '.bin' in emb_file:
    model = gensim.models.KeyedVectors.load_word2vec_format(
      emb_file, binary=True)
  else:
    model = gensim.models.Word2Vec.load(emb_file)
  sl = sorted([(k, model.vocab[k].index, model.vocab[k].count)
               for k in model.vocab.keys()],
              key=lambda (k, idx, cnt): -cnt)
  assert(select_size >= len(train_vocab))

  train_vocab = set(train_vocab) & set(model.vocab.keys())
  freq_words_num = select_size - len(train_vocab)
  if freq_words_num:
    freq_words = []
    for k, _, _ in sl:
      if k not in train_vocab:
        freq_words.append(k)
        freq_words_num -= 1
        if freq_words_num <= 0:
          break

  vocab = set(freq_words) | set(train_vocab)
  w2v = {w: model[w] for w in vocab}
  return w2v


if __name__ == '__main__':
  w2v_file = "/home/data/en_1000_no_stem/en.model"

  ascii_words = extract_ascii_words(w2v_file)
  with file('vocab_en_1000', 'w') as of:
    for k in ascii_words:
      of.write(k + '\n')

  '''w2v = read_embeddings(w2v_file, ['soccer', 'messi'], 10)
  for w in w2v:
    print w, w2v[w][:5]
  '''
