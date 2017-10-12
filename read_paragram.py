import string

valid_chars = string.ascii_letters + '_' + '.' + string.digits


def extract_ascii_words(emb_file):
  words = [line.split(' ')[0] for line in file(emb_file)]
  return [k for k in words if all(c in valid_chars for c in k)]


def read_embeddings(emb_file, train_vocab, select_size):
  vecs = {}
  for line in file(emb_file):
    tokens = line.split(' ')
    if tokens[0] not in train_vocab:
        continue
    vecs[tokens[0]] = [float(x) for x in tokens[1:]]

  emb_file2 = "/home/qv/nlp-data/paragram-phrase-XXL.txt"
  for line in file(emb_file2):
    tokens = line.split(' ')
    if tokens[0] not in train_vocab:
        continue
    vecs[tokens[0]] = [float(x) for x in tokens[1:]]
  return vecs


if __name__ == "__main__":
  emb_file = "/home/qv/nlp-data/paragram_300_sl999/paragram_300_sl999.txt"
