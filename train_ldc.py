import os
import sys
import numpy as np
import tensorflow as tf
import time
import data_utils


class FLAGS():
  pass


FLAGS.embedding_size = 300
FLAGS.embedding_type = 'GoogleNews'    # 'enwiki-skipgram'
FLAGS.w2v_file = '/home/qv/nlp-data/GoogleNews-vectors-negative300.bin'
# FLAGS.w2v_file = \
#    '/home/qv/nlp-data/enwiki-data/w2v-skipgram-300-wiki-mar-2017.txt'
# FLAGS.w2v_file = "/home/qv/nlp-data/paragram_300_sl999/paragram_300_sl999.txt"
# FLAGS.w2v_file = "/home/qv/nlp-data/glove.6B.{}d.txt".FLAGS.embedding_size


FLAGS.filter_qs_with_ans = True
FLAGS.normalize_embeddings = False

FLAGS.max_vocab_size = 100000
FLAGS.pad_word = '_NIL'
FLAGS.unknown_word = '_UNK'
FLAGS.number_word = '_number_'
FLAGS.max_s_len = 40

FLAGS.map_numbers = False
FLAGS.remove_stopwords_from_q = False
FLAGS.remove_stopwords_from_s = False

FLAGS.num_filters = 100

FLAGS.initial_learning_rate = 0.005
FLAGS.l2_reg_strength = 0.0005
FLAGS.keep_prob = 1.0
FLAGS.word_sample_prob = 1.0
FLAGS.max_gradient_norm = 100.0
FLAGS.loss = 'sigmoid_loss'
FLAGS.optimizer = 'adagrad'

FLAGS.mode = 'train'  # 'hyperparam_search' or 'train' or or 'test'

# Corpus Files
FLAGS.input_dir = '/home/qv/nlp-data/WikiQACorpus'
# Processed Data
FLAGS.data_dir = '/home/qv/wikiqa-data'
FLAGS.win_sz = 7


def train_dir_name(FLAGS):
  return \
    "{}-et={}-d={}-rsw={}-nf={}-l={}-rs={}-kp={}-ws={}".format(
      FLAGS.train_dir_base,
      FLAGS.embedding_type,
      FLAGS.embedding_size,
      FLAGS.remove_stopwords_from_s,
      FLAGS.num_filters,
      FLAGS.initial_learning_rate,
      FLAGS.l2_reg_strength,
      FLAGS.keep_prob,
      FLAGS.word_sample_prob)


FLAGS.train_dir_base = FLAGS.data_dir + '/train-ldc'
FLAGS.train_dir = train_dir_name(FLAGS)

# How often to display Training Stats
FLAGS.train_insts_bw_print = 200
FLAGS.dev_insts_bw_print = 50
FLAGS.batch_size = 10


def softmax_weighted(logits, wts=1.0, axis=-1):
    p = tf.exp(logits) * wts
    p = p / tf.expand_dims(tf.reduce_sum(p, axis), axis)
    return p


def len_mask_tbl(max_len):
  mask_tbl = np.zeros((max_len + 1, max_len), dtype=np.float32)
  for i in range(max_len + 1):
    mask_tbl[i] = [1] * i + [0] * (max_len - i)
  return mask_tbl


class CNN_LDC_Model(object):
  def __init__(self, session, data_dict, forward_only):
    self.data_dict = data_dict
    self.session = session
    i2v = data_dict['i2v']
    max_q_len = FLAGS.max_q_len  # data_dict['max_q_len']
    max_s_len = FLAGS.max_s_len  # data_dict['max_s_len']
    max_q_sents = FLAGS.max_q_sents  # data_dict['max_q_sents']

    # Debugging
    self.debug_dict = {}

    # Placeholders for input
    self.q_b = tf.placeholder(tf.int32,
                              [None, max_q_len],
                              name='questions')
    self.q_len_b = tf.placeholder(tf.int32,
                                  [None],
                                  name='question_lengths')
    self.cand_label_b = tf.placeholder(tf.int32,
                                       [None, max_q_sents],
                                       name='candiate_answer_labels')
    self.cand_ans_b = tf.placeholder(tf.int32,
                                     [None, max_q_sents, max_s_len],
                                     name='candidate_answers')
    self.cand_ans_len_b = tf.placeholder(tf.int32,
                                         [None, max_q_sents],
                                         name='candidate_answer_lengths')
    self.cand_feats_b = tf.placeholder(tf.float32,
                                       [None, max_q_sents, 2],
                                       name='candidate_features')
    self.cand_wts_b = tf.placeholder(tf.float32,
                                     [None, max_q_sents],
                                     name='candidate_ans_weights')
    self.keep_prob = tf.placeholder(tf.float32,
                                    name='keep_prob')
    self.input_keys = [self.q_b, self.q_len_b,
                       self.cand_label_b, self.cand_ans_b, self.cand_ans_len_b,
                       self.cand_feats_b, self.cand_wts_b, self.keep_prob]

    self.input_types = {k: np.int32 for k in self.input_keys}
    for k in [self.cand_feats_b, self.cand_wts_b, self.keep_prob]:
      self.input_types[k] = np.float32

    # Embedding Matrix
    nil_word_slot = tf.zeros([1, FLAGS.embedding_size])
    self.unk_word = tf.get_variable(
      'unk_word', shape=[1, FLAGS.embedding_size],
      initializer=tf.contrib.layers.xavier_initializer())
    self.number_word = tf.get_variable(
      'number_word', shape=[1, FLAGS.embedding_size],
      initializer=tf.contrib.layers.xavier_initializer())

    self.embTbl = tf.concat(0, [nil_word_slot, self.unk_word, self.number_word,
                                i2v[3:]])

    # [batch_sz, max_q_len, embedding_sz]
    S = tf.nn.embedding_lookup(self.embTbl, self.q_b)
    # [batch_sz, max_q_sents, max_q_len, embedding_sz]
    S = tf.ones([tf.shape(self.q_b)[0], FLAGS.max_q_sents,
                 FLAGS.max_q_len, FLAGS.embedding_size]) * \
        tf.expand_dims(S, 1)

    # [batch_sz, max_q_sents, max_s_len, embedding_sz]
    T = tf.nn.embedding_lookup(self.embTbl, self.cand_ans_b)

    # Dropout
    if FLAGS.keep_prob < 1.0:
        S = tf.nn.dropout(S, keep_prob=self.keep_prob)
        T = tf.nn.dropout(T, keep_prob=self.keep_prob)

    # [batch_sz, max_q_sents, max_q_len, embedding_sz]
    S_n = tf.nn.l2_normalize(S, -1)
    # [batch_sz, max_q_sents, max_s_len, embedding_sz]
    T_n = tf.nn.l2_normalize(T, -1)

    # Cosine Similarity Matrix A
    # [batch_sz, max_q_sents, max_q_len, max_s_len)
    A = tf.matmul(S_n, tf.transpose(T_n, perm=[0, 1, 3, 2]))

    qlen_mask_tbl = len_mask_tbl(FLAGS.max_q_len)
    slen_mask_tbl = len_mask_tbl(FLAGS.max_s_len)

    # [batch_sz, max_q_len]
    qlen_masks = 2 * (1 - tf.nn.embedding_lookup(qlen_mask_tbl, self.q_len_b))
    # [batch_sz, 1, max_q_len, 1]
    qlen_masks = tf.expand_dims(qlen_masks, 1)
    qlen_masks = tf.expand_dims(qlen_masks, 3)
    A = A - qlen_masks

    # [batch_sz, max_q_sents, max_s_len]
    slen_masks = 2 * (1 - tf.nn.embedding_lookup(slen_mask_tbl,
                                                 self.cand_ans_len_b))
    # [batch_sz, max_q_sents, 1, max_s_len]
    slen_masks = tf.expand_dims(slen_masks, 2)
    A = A - slen_masks

    # Semantic matching Vectors
    # [batch_sz, max_q_sents, max_q_len]
    f_match_s_T = tf.argmax(A, axis=3)
    # [batch_sz, max_q_sents, max_q_len, embedding_sz]
    S_hat = tf.matmul(tf.one_hot(f_match_s_T, FLAGS.max_s_len,
                                 on_value=1.0, off_value=0.0), T)
    # [batch_sz, max_q_sents, max_s_len]
    f_match_t_S = tf.argmax(A, axis=2)
    # [batch_sz, max_q_sents, max_s_len, embedding_sz]
    T_hat = tf.matmul(tf.one_hot(f_match_t_S, FLAGS.max_q_len,
                                 on_value=1.0, off_value=0.0), S)

    # Decomposition
    # [batch_sz, max_q_sents, max_q_len, embedding_sz]
    S_hat_n = tf.nn.l2_normalize(S_hat, -1)
    # [batch_sz, max_q_sents, max_q_len]
    S_proj = tf.reduce_sum(S * S_hat_n, -1)
    # [batch_sz, max_q_sents, max_q_len, embedding_sz]
    S_sim = tf.expand_dims(S_proj, -1) * S_hat_n
    S_desim = S - S_sim
    # [batch_sz, max_q_sents, max_q_len, embedding_sz, 2]
    S_decomp = tf.concat(4, [tf.expand_dims(S_sim, -1),
                             tf.expand_dims(S_desim, -1)])

    # [batch_sz, max_q_sents, max_s_len, embedding_sz]
    T_hat_n = tf.nn.l2_normalize(T_hat, -1)
    # [batch_sz, max_q_sents, max_s_len]
    T_proj = tf.reduce_sum(T * T_hat_n, -1)
    # [batch_sz, max_q_sents, max_s_len, embedding_sz]
    T_sim = tf.expand_dims(T_proj, -1) * T_hat_n
    T_desim = T - T_sim
    # [batch_sz, max_q_sents, max_s_len, embedding_sz, 2]
    T_decomp = tf.concat(4, [tf.expand_dims(T_sim, -1),
                             tf.expand_dims(T_desim, -1)])

    #
    # Composition
    #

    # Filter
    def compose(M_decomp, s_len, ngram_n, F=None, F_b=None):
      if not F:
        F = tf.get_variable('Filter_{}gram'.format(ngram_n),
                            [ngram_n, FLAGS.embedding_size,
                             2, FLAGS.num_filters],
                            initializer=tf.contrib.layers.xavier_initializer())
      if not F_b:
        F_b = tf.get_variable(
          'F_{}gram_bias'.format(ngram_n),
          [1, 1, 1, FLAGS.num_filters],
          initializer=tf.contrib.layers.xavier_initializer())

      # [batch_sz * max_q_sents, s_len, embedding_sz, 2]
      M_decomp = tf.reshape(M_decomp,
                            [-1, s_len, FLAGS.embedding_size, 2])
      # Convolution & Max Pooling
      # [batch_sz * max_q_sents, s_len - ngram_n + 1, 1, num_filters]
      C = tf.nn.conv2d(M_decomp, F, [1, 1, 1, 1], 'VALID') + F_b
      C = tf.tanh(C)
      C_conv = C
      
      # [batch_sz * max_q_sents, 1, 1, num_filters]
      C = tf.nn.max_pool(C, [1, s_len - ngram_n + 1, 1, 1],
                         [1, 1, 1, 1], 'VALID')

      # [batch_sz, max_q_sents, num_filters]
      C = tf.reshape(C, [-1, FLAGS.max_q_sents, FLAGS.num_filters])
      return (F, F_b, C, C_conv)

    # Compose unigram, bigram & trigram features from question
    Fu, Fu_b, S_uni, Su_c = compose(S_decomp, FLAGS.max_q_len, 1)
    Fb, Fb_b, S_bi, Sb_c = compose(S_decomp, FLAGS.max_q_len, 2)
    Ft, Ft_b, S_tri, St_c = compose(S_decomp, FLAGS.max_q_len, 3)

    # Compose unigram, bigram & trigram features from sentence
    _, _, T_uni, Tu_c = compose(T_decomp, FLAGS.max_s_len, 1, Fu, Fu_b)
    _, _, T_bi, Tb_c = compose(T_decomp, FLAGS.max_s_len, 2, Fb, Fb_b)
    _, _, T_tri, Tt_c = compose(T_decomp, FLAGS.max_s_len, 3, Ft, Ft_b)

    # [batch_sz, max_q_sents, 6 * FLAGS.num_filters]
    feats = tf.concat(2, [S_uni, S_bi, S_tri, T_uni, T_bi, T_tri])
    
    self.W = tf.get_variable('sim_weights',
                             [1, 1, 6 * FLAGS.num_filters],
                             initializer=tf.contrib.layers.xavier_initializer())
    self.W_b = tf.get_variable(
      'sim_weights_bias', [1, 1],
      initializer=tf.contrib.layers.xavier_initializer())

    # [batch_sz, max_q_sents]
    logits = tf.reduce_sum(feats * self.W, -1) + self.W_b
    self.pred = tf.maximum(tf.minimum(tf.sigmoid(logits), 0.9999), 0.0001)
    # self.pred = tf.sigmoid(logits)

    if FLAGS.loss == 'sigmoid_loss':
      label = tf.cast(self.cand_label_b, tf.float32)
      loss = -1.0 * (label * tf.log(self.pred) +
                     (1 - label) * tf.log(1 - self.pred))
    else:
      assert(False)

    # Sum the loss across the whole batch
    loss = loss * self.cand_wts_b
    self.loss = tf.reduce_sum(loss) / tf.reduce_sum(self.cand_wts_b)

    params = tf.trainable_variables()
    if FLAGS.l2_reg_strength > 0:
      '''self.regularization_loss = FLAGS.l2_reg_strength * \
          tf.nn.l2_loss(self.W)
      self.loss = self.loss + self.regularization_loss
      '''
      self.regularization_loss = FLAGS.l2_reg_strength * \
          tf.add_n([tf.nn.l2_loss(p) for p in params])
      self.loss = self.loss + self.regularization_loss

    else:
      self.regularization_loss = tf.constant(0)

    #
    # Predictions
    #

    # [batch_sz]
    # self.pred = tf.argmax(logits, 1)

    # Gradients
    if not forward_only:
      if FLAGS.optimizer == 'adagrad':
        self.opt = tf.train.AdagradOptimizer(FLAGS.initial_learning_rate)
      elif FLAGS.optimizer == 'adam':
        self.opt = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)
      elif FLAGS.optimizer == 'gd':
        self.opt = tf.train.GradientDescentOptimizer(
          FLAGS.initial_learning_rate)
      elif FLAGS.optimizer == 'lbfgs':
        self.opt = tf.contrib.opt.ScipyOptimizerInterface(
          self.loss, options={'maxiter': 100})
      else:
        assert(False)

      if FLAGS.optimizer != 'lbfgs':
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_gradient_norm), va)
                          for g, va in grads_and_vars if g is not None]

        self.update = self.opt.apply_gradients(grads_and_vars, name='train_op')

    self.saver = tf.train.Saver(tf.global_variables())

    # Debugging
    self.debug_dict['A'] = A
    self.debug_dict['f_match'] = (f_match_s_T, f_match_t_S)
    self.debug_dict['decomp'] = (S_proj, S_sim, S_desim, T_proj, T_sim, T_desim)
    self.debug_dict['F'] = (Fu, Fb, Ft, Fu_b, Fb_b, Ft_b)
    self.debug_dict['conv'] = (Su_c, Sb_c, St_c, Tu_c, Tb_c, Tt_c)
    self.debug_dict['feats'] = feats
    self.debug_dict['W'] = (self.W, self.W_b)
    self.debug_dict['sim'] = self.pred
    self.debug_dict['loss'] = self.loss

  def normalize_embeddings(self):
    if FLAGS.normalize_embeddings:
      self.session.run(self.normalize_A)

  def step(self, inputs,
           forward_only, test_only=False):
    if not forward_only:
      if FLAGS.optimizer == 'lbfgs':
        self.opt.minimize(self.session, inputs)
        output_feed = [self.regularization_loss, self.loss, self.F]
      else:
        output_feed = [self.regularization_loss, self.loss,
                       self.pred, self.update]
      self.normalize_embeddings()
    elif not test_only:
      output_feed = [self.regularization_loss, self.loss, self.pred]
    else:
      output_feed = [self.pred]

    outputs = self.session.run(output_feed, inputs)
    return outputs


def questions_with_answers(q_to_s_map):
    return [qn for qn, slist in q_to_s_map.items()
            if any(1 if l == 1 else 0 for _, l, _, _ in slist)]


class data_set():
  def __init__(self, model, data_set_str, qnum=None,
               forward_only=False, test_only=False):
    self.model = model
    self.forward_only = forward_only
    self.test_only = test_only
    data_dict = model.data_dict
    self.data_set_str = data_set_str
    self.q_to_s_map = data_dict['q_to_s_map'][data_set_str]
    self.v_qs = data_dict['v_qs'][data_set_str]
    self.v_sentences = data_dict['v_sentences'][data_set_str]
    self.q_to_kv_map = data_dict['q_to_kv_maps'][data_set_str]
    self.qnum = qnum

    if qnum:
      idxs = [qnum]
      self.batch_size = 1
    else:
      self.batch_size = FLAGS.batch_size
      if not FLAGS.filter_qs_with_ans:
        idxs = self.q_to_s_map.keys()
      else:
        idxs = questions_with_answers(self.q_to_s_map)
    self.data_idx_set = idxs
    self.tokens = self.generate_tokens()

  def generate_tokens(self):
    # Get the # of batches from data_idx_set and create a turn token for each
    # batch.  The token identifies the batch's bucket and batch # within the
    # bucket.
    tokens = []
    data_set_sz = len(self.data_idx_set)
    num_batches = data_set_sz / self.batch_size
    for batch_id in range(num_batches):
        tokens.append(batch_id)
    return tokens

  def start(self):
    # Shuffle the data indices
    np.random.shuffle(self.data_idx_set)
    self.batches_processed = 0

  def done(self):
    return self.batches_processed >= len(self.tokens)

  def prepare_q_data(self, q_num):

    def random_select(seq):
      if self.forward_only or not FLAGS.word_sample_prob:
        return seq
      p = FLAGS.word_sample_prob
      selected = np.random.choice([0, 1], size=len(seq), p=[1 - p, p])
      ret = [s for i, s in enumerate(seq) if selected[i]]
      return ret if len(ret) else seq

    def pad_seq(seq, max_len):
      return seq[:max_len] + [0] * (max_len - len(seq))

    pad_sentence = [0] * FLAGS.max_s_len

    q = random_select(self.v_qs[q_num])
    q_len = len(q)
    q = pad_seq(q, FLAGS.max_q_len)
    slist = self.q_to_s_map[q_num]
    cand_ans_labels = [l for s_id, l, _, _ in slist]
    cand_ans_labels += [0] * (FLAGS.max_q_sents - len(slist))
    cand_ans = [random_select(self.v_sentences[s_id])
                for s_id, _, _, _ in slist]
    cand_ans_len = [min(len(ca), FLAGS.max_s_len) for ca in cand_ans]
    cand_ans_len += [1] * (FLAGS.max_q_sents - len(cand_ans_len))
    cand_ans = [pad_seq(ca, FLAGS.max_s_len) for ca in cand_ans]
    cand_ans += [pad_sentence] * (FLAGS.max_q_sents - len(cand_ans))
    cand_features = [[cnt, idf_wtd_cnt]
                     for s_id, _, cnt, idf_wtd_cnt in slist]
    cand_features += [[0.0, 0.0]] * (FLAGS.max_q_sents -
                                     len(cand_features))
    cand_wts = [1] * len(slist) + [0] * (FLAGS.max_q_sents - len(slist))
    model = self.model
    return {model.q_b: q, model.q_len_b: q_len,
            model.cand_label_b: cand_ans_labels, model.cand_ans_b: cand_ans,
            model.cand_ans_len_b: cand_ans_len,
            model.cand_feats_b: cand_features,
            model.cand_wts_b: cand_wts}

  def prepare_batch(self, batch_num):
    model = self.model
    batch_q_nums = []
    batch_inputs = {k: [] for k in model.input_keys if k != model.keep_prob}

    start_idx = batch_num * self.batch_size
    for offset in xrange(self.batch_size):
      q_num = self.data_idx_set[start_idx + offset]
      inputs = self.prepare_q_data(q_num)
      batch_q_nums.append(q_num)
      for k in batch_inputs.keys():
        batch_inputs[k].append(inputs[k])

    feed_dict = {k: np.array(batch_inputs[k], dtype=model.input_types[k])
                 for k in batch_inputs.keys()}
    keep_prob = FLAGS.keep_prob if not self.forward_only else 1.0
    feed_dict[model.keep_prob] = keep_prob
    return (batch_q_nums, feed_dict)

  def q_to_ans_idx(self, q_num):
      slist = self.q_to_s_map[q_num]
      ans_idx = [i for i, (s_id, l, _, _) in enumerate(slist) if l == 1][0]
      return ans_idx

  def answer_rank(self, q_num, s_probs):
    slist = self.q_to_s_map[q_num]
    ans_idxs = [i for i, (s_id, l, _, _) in enumerate(slist) if l == 1]
    if not ans_idxs:
      return None, None

    if FLAGS.loss == 'softmax_loss':
      ranks = sorted([(prob[1], idx) for idx, prob in enumerate(s_probs)],
                     key=lambda (p, i): -p)
    else:
      ranks = sorted([(prob, idx) for idx, prob in enumerate(s_probs)],
                     key=lambda (p, i): -p)

    min_rank = len(ranks) + 1
    avp = 0
    ans_cnt = 0.0
    for r, (_, j) in enumerate(ranks, 1):
      if j in ans_idxs:
        if r < min_rank:
          min_rank = r
          s_off = j
        ans_cnt += 1.0
        avp += ans_cnt / r

    return min_rank, s_off, avp / len(ans_idxs)

  def predictions(self, q_num, s_probs):
    slist = self.q_to_s_map[q_num]
    preds = {}
    corr_cnt = 0
    tp = 0
    fp = 0
    for i, prob in enumerate(s_probs):
      s_id, l, _, _ = slist[i]
      if FLAGS.loss == 'softmax_loss':
        prob = prob[1]

      if (prob > 0.5 and l or prob <= 0.5 and not l):
        preds[s_id] = (1, l)
        corr_cnt += 1
        if l == 1:
          tp += 1
      else:
        preds[s_id] = (0, l)
        if l != 1:
          fp += 1
    return preds, corr_cnt, tp, fp

  def print_batch(self, batch_num, probs):

    start_idx = batch_num * self.batch_size
    for offset in xrange(self.batch_size):
      q_num = self.data_idx_set[start_idx + offset]
      q = self.v_qs[q_num]

      print '=> Question #:', q_num
      print ' '.join([self.rev_vocab[t] for t in q if t != 0])
      print "**Predicted Answer**:"
      preds, _ = self.predictions(q_num, probs[offset])
      for s_id, (corr, l) in preds.items():
        sentence = self.v_sentences[s_id]
        print ' '.join([self.rev_vocab[t] for t in sentence if t != 0])
        print "  prediction: {}, label: {}".format(
          "correct" if corr else "*incorrect*", l)

  def run_batches(self, batch_count=None):
    # Run the epoch by looping over the tokens
    loss = 0.0
    reg_loss_total = 0.0
    mrr = 0.0
    map = 0.0
    instance_count = 0
    correct_predictions = 0
    mrr_count = 0
    bad_q_nums = []
    tps, fps = (0, 0)
    top = 0

    if not batch_count:
      batch_count = len(self.tokens)

    start = self.batches_processed
    end = self.batches_processed + batch_count
    self.batches_processed = end
    for batch_num in self.tokens[start:end]:
      q_nums, inputs = self.prepare_batch(batch_num)

      if not self.test_only:
        reg_loss, step_loss, probs = \
            self.model.step(inputs, self.forward_only, self.test_only)[:3]
        loss += step_loss
        reg_loss_total += reg_loss
      else:
        probs = self.model.step(inputs,
                                self.forward_only, self.test_only)[0]
        if FLAGS.mode == 'test_debug':
          self.print_batch(batch_num, probs)

      for i, q_num in enumerate(q_nums):
        instance_count += len(self.q_to_s_map[q_num])

      if not self.forward_only:
        continue

      for i, q_num in enumerate(q_nums):
        s_cnt = len(self.q_to_s_map[q_num])
        s_probs = probs[i][:s_cnt]
        _, corr_cnt, tp, fp = self.predictions(q_num, s_probs)
        correct_predictions += corr_cnt
        tps += tp
        fps += fp

        rank, s_off, avp = self.answer_rank(q_num, s_probs)
        if rank:
          map += avp
          mrr = mrr + 1.0 / rank
          mrr_count += 1
          if rank != 1:
            bad_q_nums.append(q_num)
          else:
            top += 1
          # print '''q_num {}: sentence offset {} rank {} mrr {} count {},
          #         top {}'''.format(
          #    q_num, s_off, rank, mrr, mrr_count, top)

    # print "TP: {}, FP: {}".format(tps, fps)
    avg_loss = 1.0 * loss / instance_count
    reg_loss_avg = 1.0 * reg_loss_total / instance_count
    if mrr_count:
      mrr = mrr / mrr_count
      map = map / mrr_count
    return (reg_loss_avg, avg_loss, tps, fps, top,
            mrr, map, correct_predictions, instance_count, bad_q_nums)


def create_model(session, data_dict, forward_only):
  # Create model.
  print("Creating Key-Value End-to-End Memory Network Model")
  model = CNN_LDC_Model(session, data_dict, forward_only)

  # Merge all the summaries and write them out to /tmp/train (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train',
                                        graph=session.graph)
  #test_writer = tf.summary.FileWriter(FLAGS.train_dir + '/test')

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    model.normalize_embeddings()

  return model


def train(data_dict, epochs=500):

  dev_b_cnt = FLAGS.dev_insts_bw_print / FLAGS.batch_size
  train_b_cnt = FLAGS.train_insts_bw_print / FLAGS.batch_size

  best_results = None
  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = create_model(session, data_dict, False)
    train_set = data_set(model, 'train')
    dev_set = data_set(model, 'dev', forward_only=True)

    # This is the training loop.
    best_mrr = 0.0
    for epoch in range(epochs):

      print 'Epoch: ', epoch
      train_set.start()

      while True:
        start_time = time.time()
        reg_loss, loss, _, _, _, _, _, _, _, _ = train_set.run_batches(train_b_cnt)

        run_time = time.time() - start_time

        # Print statistics for the previous epoch.
        print("run-time %.2f cross-entropy loss %.4f reg-loss %.4f" %
              (run_time, loss, reg_loss))

        dev_set.start()
        reg_loss, loss, tps, fps, top, mrr, map, cps, n, _ = \
            dev_set.run_batches(dev_b_cnt)
        print(" eval: reg-loss %.4f loss %.4f tps/fps %d/%d tops %d "
              "mrr %.2f map %.2f corr preds %d/%d" %
              (reg_loss, loss, tps, fps, top, mrr, map, cps, n))

        if train_set.done():
          break

      # Print statistics for the previous epoch.
      # print "epoch-time %.2f cross-entropy loss %.2f" % (epoch_time, loss)
      dev_set.start()
      reg_loss, loss, tps, fps, top, mrr, map, cps, n, _ = \
          dev_set.run_batches()
      print(" eval: reg-loss %.4f loss %.4f tps/fps %d/%d tops %d "
            "mrr %.2f map %.2f corr preds %d/%d" %
            (reg_loss, loss, tps, fps, top, mrr, map, cps, n))

      # Save checkpoint and zero timer and loss.
      if mrr > best_mrr:
        best_mrr = mrr
        best_results = (reg_loss, loss, tps, fps, top, mrr, map, cps, n)
        checkpoint_path = os.path.join(FLAGS.train_dir, "wikiqa.ckpt")
        model.saver.save(session, checkpoint_path)

      # Run evals on development set and print their perplexity.
      # eval_loss, accuracy = run_epoch(sess, model, dev_set, True)
      # print("  eval: loss %.2f, accuracy %.2f" % (eval_loss, accuracy))
      sys.stdout.flush()

    return best_results


def test(data_dict, batch_count=None, data_set_str='test', q_num=None):
  with tf.Session() as session:
    with tf.device('/cpu:0'):
      model = create_model(session, data_dict, True)
    test_set = data_set(model, data_set_str, qnum=q_num,
                        forward_only=True, test_only=True)

    test_set.start()
    _, _, tps, fps, top, mrr, map, cps, n, bad_q_nums = \
        test_set.run_batches(batch_count)
    print(" %s: tps/fps %d/%d tops %d mrr %.2f map %.2f corr preds %d/%d" %
          (data_set_str, tps, fps, top, mrr, map, cps, n))
    print len(test_set.data_idx_set)
    sys.stdout.flush()

  return bad_q_nums


def hyperparam_search(FLAGS, override=False):
    t1 = time.time()
    ofile = open('/home/qv/wikiqa-data/out.train_ldc.txt', 'a')
    i = 0
    while True:
      FLAGS.initial_learning_rate = np.random.choice([0.05, 0.005])
      FLAGS.l2_reg_strength = np.random.choice([0.005, 0.0005])
      FLAGS.keep_prob = np.random.choice([1, 0.75, 0.5])
      FLAGS.embedding_type = np.random.choice(['enwiki-skipgram', 'GoogleNews'])
      FLAGS.remove_stopwords_from_s = np.random.choice([True, False])
      FLAGS.num_filters = np.random.choice([10, 100, 500])
      FLAGS.train_dir = train_dir_name(FLAGS)
      FLAGS.data_pkl_file = data_utils.processed_data_file_name(FLAGS)

      if os.path.exists(FLAGS.train_dir) and not override:
        continue

      data_dict = data_utils.prepare_data(FLAGS)
      FLAGS.max_q_sents = data_dict['max_q_sents']
      FLAGS.max_q_len = data_dict['max_q_len']

      epochs = 100
      if FLAGS.initial_learning_rate <= 0.01:
        epochs = 200
      print "Learning rate: ", FLAGS.initial_learning_rate
      tf.reset_default_graph()
      best_results = train(data_dict, epochs)

      t = (FLAGS.initial_learning_rate, FLAGS.l2_reg_strength, FLAGS.keep_prob,
           FLAGS.embedding_type, FLAGS.embedding_size,
           FLAGS.remove_stopwords_from_s, FLAGS.num_filters)
      opt_names = ['initial_learning_rate', 'l2_reg_strength', 'keep_prob',
                   'embedding_type', 'embedding_size', 'stopwords removed',
                   'num_filters']
      ofile.write(FLAGS.train_dir + '\n')
      ofile.write(FLAGS.data_pkl_file + '\n')
      ofile.write('Parameters: \n')
      for name, opt in zip(opt_names, t):
        ofile.write('  {}: {}\n'.format(name, opt))
      ofile.write('Best Results: ')
      ofile.write("  reg-loss %.4f loss %.4f tps/fps %d/%d tops %d\n"
                  "  mrr %.2f map %.2f corr preds %d/%d\n" % best_results)
      ofile.write('\n')
      ofile.flush()

      i = i + 1
      if i == 100:
        break

    ofile.close()
    print time.time() - t1
  

if __name__ == "__main__":

  # FLAGS.mode = 'test' #'hyperparam_search'  # 'train' or 'test_debug' or 'test'

  '''if FLAGS.mode == 'test':
    FLAGS.initial_learning_rate = 0.005
    FLAGS.l2_reg_strength = 0
    FLAGS.keep_prob = 1.0
    FLAGS.embedding_type = 'enwiki-skipgram'  # 'enwiki-skipgram'
    FLAGS.remove_stopwords_from_s = False
    FLAGS.num_filters = 100
    FLAGS.train_dir = train_dir_name(FLAGS)'''

  if FLAGS.mode is not 'hyperparam_search':
    FLAGS.data_pkl_file = data_utils.processed_data_file_name(FLAGS)
    data_dict = data_utils.prepare_data(FLAGS)
    FLAGS.max_q_sents = data_dict['max_q_sents']
    FLAGS.max_q_len = data_dict['max_q_len']

  if FLAGS.mode == 'test':
    test(data_dict, batch_count=None, data_set_str='test')
  elif FLAGS.mode == 'test_debug':
    q_num = 1
    test(data_dict, batch_count=None, data_set_str='dev', q_num=q_num)
  elif FLAGS.mode is 'hyperparam_search':
    hyperparam_search(FLAGS)
  elif FLAGS.mode is None or FLAGS.mode == 'train':
    print "Learning rate: ", FLAGS.initial_learning_rate
    train(data_dict)
  else:
    assert(False)
