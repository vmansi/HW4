from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

import math
import os
import random
import sys
import time
import logging
import pickle as pk
import string
from nltk import word_tokenize
import math
from operator import itemgetter

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
						  "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
						  "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
							"Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "Data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "Checkpoints", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
							"Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
							"How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
							"Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
							"Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
							"Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)


def get_most_similar_sent(unigram_table, bigram_table, dialog_list, vocab_len, s2):
	prob_list = []
	tokens_s2 = word_tokenize(s2)

	for i,dialog in enumerate(dialog_list):
		prob_list.append([i,0])
		len_dialog = len(dialog)
		for l,word1 in enumerate(word_tokenize(dialog)):
			# temp_prob_list = []
			if unigram_table.get(word1):
				den = unigram_table.get(word1) + vocab_len*0.25
				prob_list[i][1] = prob_list[i][1] + math.log(unigram_table[word1]/2042301.0, 2)
			else:
				den = vocab_len*0.25		
			for word2 in tokens_s2:
				if bigram_table.get((word1, word2)):
					num = bigram_table[(word1, word2)] + 0.25
					#print num
				else:
					num = 0.25

			prob_list[i][1] = prob_list[i][1] + math.log(num/float(den), 2)

		prob_list[i][1] = prob_list[i][1]/float(len_dialog**1)	
	prob_list.sort(key=lambda tup: tup[1], reverse = True)  # sorts in place
	idx = random.randint(0, 2)
	print idx
	return dialog_list[prob_list[idx][0]]

def create_model(session, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	model = seq2seq_model.Seq2SeqModel(
	  FLAGS.from_vocab_size,
	  FLAGS.to_vocab_size,
	  _buckets,
	  FLAGS.size,
	  FLAGS.num_layers,
	  FLAGS.max_gradient_norm,
	  FLAGS.batch_size,
	  FLAGS.learning_rate,
	  FLAGS.learning_rate_decay_factor,
	  forward_only=forward_only,
	  dtype=dtype)
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir, latest_filename = "checkpoint")
	print (ckpt)
	#read model
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
		return model
	else:
		print ("Model doesnot exist")

sess = tf.Session()
filter_unigram_table = pk.load(open('filter_unigram_table.p', 'rb'))
filter_bigram_table = pk.load(open('filter_bigram_table.p', 'rb'))

f = open('starttrekDialogues.txt',"r")
dialogs = f.readlines()
f.close()

dialog_list = [dialog.strip() for dialog in dialogs]


model = create_model(sess, True)
model.batch_size = 1  # We decode one sentence at a time.

	# Load vocabularies.
en_vocab_path = os.path.join(FLAGS.data_dir,
							 "vocab%d.from" % FLAGS.from_vocab_size)
fr_vocab_path = os.path.join(FLAGS.data_dir,
							 "vocab%d.to" % FLAGS.to_vocab_size)
en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)


@ask.launch
def launch_alexa():
	welcome_msg = 'welcome to twitter bot.'
	return statement(welcome_msg)

@ask.intent("ChatIntent", convert = {"Text" : str})
def chat_alexa(Text):
	print ("Text")
	print (Text)
	token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(Text), en_vocab)
	# Which bucket does it belong to?
	bucket_id = len(_buckets) - 1
	for i, bucket in enumerate(_buckets):
	  if bucket[0] >= len(token_ids):
		bucket_id = i
		break
	else:
	  logging.warning("Sentence truncated: %s", Text)

	# Get a 1-element batch to feed the sentence to the model.
	encoder_inputs, decoder_inputs, target_weights = model.get_batch(
		{bucket_id: [(token_ids, [])]}, bucket_id)
	# Get output logits for the sentence.
	_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
									 target_weights, bucket_id, True)
	# This is a greedy decoder - outputs are just argmaxes of output_logits.
	outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
	# If there is an EOS symbol in outputs, cut them at that point.
	if data_utils.EOS_ID in outputs:
	  outputs = outputs[:outputs.index(data_utils.EOS_ID)]
	# Print out French sentence corresponding to outputs.
	reply = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
	prev_sent = get_most_similar_sent(filter_unigram_table, filter_bigram_table, dialogs, 44332, reply.translate(None, string.punctuation)
)
	
	print ("reply")
	print (prev_sent+ " "+ reply)
	return statement(prev_sent+ " "+ reply)		
	# reply = translate.decode(Text)
	# return statement(reply)
	# return statement(Text)    


if __name__ == '__main__':
	app.run(debug=True)
