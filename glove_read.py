import string

valid_chars = string.ascii_letters + '_' + '.' + string.digits


def extract_ascii_words(emb_file):
  words = [line.split()[0] for line in file(emb_file)]
  return [k for k in words if all(c in valid_chars for c in k)]


def read_embeddings(emb_file, train_vocab, select_size):
  gl_words = [line.split()[0] for line in file(emb_file)]
  gl_vecs = [[float(x) for x in line.split()[1:]]
             for line in file(emb_file)]
  model = {w: v for w, v in zip(gl_words, gl_vecs)}
  assert(select_size >= len(train_vocab))

  train_vocab = set(train_vocab) & set(gl_words)
  freq_words_num = select_size - len(train_vocab)
  if freq_words_num:
    freq_words = []
    for k in gl_words:
      if k not in train_vocab:
        freq_words.append(k)
        freq_words_num -= 1
        if freq_words_num <= 0:
          break

  vocab = set(freq_words) | set(train_vocab)
  w2v = {w: model[w] for w in vocab}
  return w2v


if __name__ == '__main__':
  w2v_file = '/home/qv/nlp-data/glove.6B.50d.txt'

  w2v = read_embeddings(w2v_file, ['soccer', 'messi'], 10)
  for w in w2v:
    print w, w2v[w][:5]
