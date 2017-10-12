#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import collections
import train_ldc2 as train
import data_utils

rev_vocab = None
data_dict = None


class DEBUG:
  pass


DEBUG.level = 0
FLAGS = None

def vector_to_text(v):
  return [rev_vocab[t] for t in v if t != 0]


def s_offset_to_s_id(q_to_s_map, qnum, s_offset):
  slist = q_to_s_map[qnum]
  if s_offset < len(slist):
    return slist[s_offset][0]
  else:
    return None


def cmp_q_to_token_list(i2v, q_e, tokens):
    for t_id in tokens:
      t_e = i2v[t_id]
      cs = cosine_similarity(t_e.reshape(1, -1), q_e.reshape(1, -1))
      dp = np.dot(t_e.reshape(1, -1), q_e.reshape(-1, 1))
      t = rev_vocab[t_id]
      print '    norm({}): {}, dotp(q, {}): {}, cos-sim(q, {}): {}'.format(
        t, np.linalg.norm(t_e), t, dp, t, cs)
      print


def show_qa(qnum, data_dict, data_set_str, i2v, q_input, q_e):
  v_qs = data_dict['v_qs'][data_set_str]

  q = v_qs[qnum]
  print "Question: ", qnum
  print '  ', vector_to_text(q_input)
  if DEBUG.level == 2:
    print '  ', q
    print '   actual input:'
    print '     q: ', list(q_input[:12])
    print '     q norm: ', np.linalg.norm(q_e)
    cmp_q_to_token_list(i2v, q_e, q)


def print_sentence(i, s, s_id, l,
                   cand_ans, cand_label, cand_ans_e):
  print 'Candidate Answer: ', i
  print '  ', vector_to_text(cand_ans[:20])
  print '   sid: {}, label: {}'.format(s_id, cand_label)
  if any(1 for j in range(len(s[:FLAGS.max_s_len])) if s[j] != cand_ans[j]):
    print '   mismatch b/w actual input and tf input'
    print '    ', vector_to_text(s[:20])
    print
  if l != cand_label:
    print '  ', 'mismatch b/w actual input and tf input'
    print '  ', '  actual label: ', l

  if DEBUG.level == 2:
    print '   sentence norm: ', np.linalg.norm(cand_ans_e[i])


def debug_question(model, qnum, data_set_str, i2v):
  # w_d_freq, w_val_freq, w_q_freq, w_ans_freq = w_freq_stats
  data_dict = model.data_dict
  q_to_s_map = data_dict['q_to_s_map'][data_set_str]
  debug_set = train.data_set(model, data_set_str, qnum=qnum,
                             forward_only=True)

  debug_set.start()
  q_nums, inputs = debug_set.prepare_batch(0)
  debug_obj_names = model.debug_dict.keys()
  debug_objs = model.debug_dict.values()
  results = model.session.run(debug_objs, inputs)
  debug_results = {k: r for k, r in zip(debug_obj_names, results)}

  q_inputs = inputs[model.q_b][0]
  cand_labels = inputs[model.cand_label_b][0]
  cand_ans = inputs[model.cand_ans_b][0]

  S_proj, S_sim, S_desim, T_proj, T_sim, T_desim = debug_results['decomp']
  q_e = S_sim[0][0] + S_desim[0][0]
  show_qa(qnum, data_dict, data_set_str, i2v, q_inputs, q_e)
  print

  loss = debug_results['loss']
  print "loss: ", loss
  sim = debug_results['sim']
  print "similarity: ", sim

  feats = debug_results['feats']
  W, W_b = debug_results['W']
  Su_c, Sb_c, St_c, Tu_c, Tb_c, Tt_c = debug_results['conv']
  convs = {('q', 'uni'): Su_c, ('q', 'bi'): Sb_c, ('q', 'tri'): St_c,
           ('ans', 'uni'): Tu_c, ('ans', 'bi'): Tb_c, ('ans', 'tri'): Tt_c}
  f_match_s_T, f_match_t_S = debug_results['f_match']
  cand_ans_e = T_sim[0] + T_desim[0]

  def ngram_details(w_idx, ngram_type, seq, other_seq, s_off):
    ngram_n = ngram_type % 3 + 1
    semantic_map = []
    ngram_words = []
    f_match = f_match_s_T if ngram_type < 3 else f_match_t_S
    for k in range(w, w + ngram_n):
      if k >= len(seq):
        ngram_words.append(rev_vocab[0])
        semantic_map.append(rev_vocab[0])
      else:
        best_match_w = f_match[0][s_off][k]
        ngram_words.append(rev_vocab[seq[k]])
        semantic_map.append(rev_vocab[other_seq[best_match_w]])
    return '{} -> {}'.format(ngram_words, semantic_map)

  slist = q_to_s_map[qnum]
  q = data_dict['v_qs'][data_set_str][qnum]
  for s_off, (s_id, l, cnt, wtd_cnt) in enumerate(slist):

    sent = data_dict['v_sentences']['dev'][s_id]
    print_sentence(s_off, sent, s_id, l,
                   cand_ans[s_off], cand_labels[s_off], cand_ans_e[s_off])

    print '  Best maches for q words:'
    print '   ',
    print ', '.join(['{} -> {}'.format(
      rev_vocab[w_id],
      rev_vocab[sent[f_match_s_T[0][s_off][pos]]])
      for pos, w_id in enumerate(q)])
    print '  Best maches for sentence words:'
    print '   ',
    print ', '.join(['{} -> {}'.format(
      rev_vocab[w_id],
      rev_vocab[q[f_match_t_S[0][s_off][pos]]])
      for pos, w_id in enumerate(sent[:FLAGS.max_s_len])])

    feat_contribs = feats[0][s_off] * W[0][0]
    logit = sum(feat_contribs)

    ng_contribs = {}
    for f_idx in range(6 * train.FLAGS.num_filters):
      ng_type = f_idx / train.FLAGS.num_filters
      ngram_n = ng_type % 3 + 1
      ngram = ['uni', 'bi', 'tri'][ngram_n - 1]
      q_or_a = 'q' if (f_idx < 3 * train.FLAGS.num_filters) else 'ans'
      f_num = f_idx % train.FLAGS.num_filters
      conv = convs[(q_or_a, ngram)]
      w_idx = np.argmax(conv[s_off, :, 0, f_num])
      ng_contribs[(w_idx, ng_type)] = ng_contribs.get((w_idx, ng_type), 0) + \
          feat_contribs[f_idx]

    imp_ngs = sorted([(w, n, contrib)
                      for (w, n), contrib in ng_contribs.items()],
                     key=lambda (w, n, c): -np.abs(c))[:10]

    # Find the top 10 contributors of similarity
    print '  Most important ngrams:'
    for w, ng_type, contrib in imp_ngs:
      print '   ',
      q_or_a = 'q' if ng_type < 3 else 'ans'
      seq = q if q_or_a is 'q' else sent
      other_seq = sent if q_or_a is 'q' else q
      print '    ',
      print '{} ngram {} contributes {} (out of {})'.format(
        q_or_a, ngram_details(w, ng_type, seq, other_seq, s_off),
        contrib, logit)


def debug_questions(data_dict, data_set_str, q_num_set):
  # w_f_stats = w_freq_stats(data_dict)
  tf.reset_default_graph()
  with tf.Session() as session:
    with tf.device('/cpu:0'):
      model = train.create_model(session, data_dict, True)
    i2v = session.run([model.embTbl])[0]
    for qn in q_num_set:
      debug_question(model, qn, data_set_str, i2v)  # , w_f_stats)
      print(u'=' * 80)


if __name__ == "__main__":
  FLAGS = train.FLAGS
  FLAGS.initial_learning_rate = 0.05
  FLAGS.l2_reg_strength = 0.0005
  FLAGS.keep_prob = 1.0
  FLAGS.embedding_type = 'enwiki-skipgram'
  FLAGS.remove_stopwords_from_s = False
  FLAGS.num_filters = 100
  FLAGS.train_dir = train.train_dir_name(FLAGS)
  FLAGS.data_pkl_file = data_utils.processed_data_file_name(FLAGS)
  
  data_dict = data_utils.prepare_data(FLAGS)
  train.FLAGS.max_q_sents = data_dict['max_q_sents']
  train.FLAGS.max_q_len = data_dict['max_q_len']

  rev_vocab = data_dict['rev_vocab']
  q_num_set = train.questions_with_answers(data_dict['q_to_s_map']['dev'])
  # q_num_set = np.random.permutation(q_num_set)
  # q_num_set = set([112])
  debug_questions(data_dict, 'dev', q_num_set)

  '''
  bad_q_nums = train.test(data_dict, batch_count=None, data_set_str='dev')
  bad_qn_sample = np.random.choice(bad_q_nums, 100, replace=False)
  debug_questions(data_dict, bad_qn_sample)
  good_q_nums = set(data_dict['v_qa_pairs_data'][1].keys()) - set(bad_q_nums)
  good_qn_sample = np.random.choice(list(good_q_nums), 100, replace=False)
  debug_questions(data_dict, good_qn_sample)
  '''
