from JeopardyNetwork import JeopardyNetwork
import tensorflow as tf
import data_utils
import numpy as np
import sys
import time
import os
import math


def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(token_ids)])

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    return " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.
    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
          source, target = source_file.readline(), target_file.readline()
          counter = 0
          while source and target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
              if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids])
                break
            source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only, checkpoint_dir):

    """Create model and initialize or load parameters"""
    model = JeopardyNetwork(
      source_vocab_size, target_vocab_size, buckets, size,
      num_layers, max_gradient_norm, batch_size, learning_rate,
      use_lstm, num_layers, forward_only)

    if os.path.exists(checkpoint_dir):
        model.saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        print("Model was restored.")
    else:
        os.makedirs(checkpoint_dir, exist_ok=False)

    session.run(tf.global_variables_initializer())
    print("Variables initialized...")
    return model


def train_model(checkpoint_dir):
    with tf.Session() as sess:
        model = create_model(sess, False, checkpoint_dir)
        sess.run(tf.initialize_all_variables())
        print("Variables initialized...")

        enc_train, dec_train, _, _ = data_utils.prepare_custom_data(
            data_dir, encoding_file, decoding_file, source_vocab_size, target_vocab_size)
        print("Data encoded...")
        train_set = read_data(enc_train, dec_train, max_size=None)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        print("Starting training...")
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS_steps_per_checkpoint
            loss += step_loss / FLAGS_steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS_steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), step_time, perplexity))
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(checkpoint_dir, "seq2seq.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()


def decode(checkpoint_dir):
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True, checkpoint_dir)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        enc_vocab_path = os.path.join("./Runs_50k_20k_vocab/","vocab%d.enc" % source_vocab_size)
        dec_vocab_path = os.path.join("./Runs_50k_20k_vocab/","vocab%d.dec" % target_vocab_size)

        enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
        _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(sentence, enc_vocab)
            # Which bucket does it belong to?
            bucket_id = min([b for b in range(len(buckets))
                           if buckets[b][0] > len(token_ids)])
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
            # Print out the question that would be asked
            print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


if __name__ == '__main__':

    data_dir = "./Runs_50k_20k_vocab/"
    encoding_file = "./Runs_50k_20k_vocab/story_text_per_line_lt65"
    decoding_file = "./Runs_50k_20k_vocab/question_per_line_lt65"

    FLAGS_steps_per_checkpoint = 200
    FLAGS_should_load_model = True

    source_vocab_size = 50000
    target_vocab_size = 20000
    buckets = [(20, 5), (30, 6), (55, 8), (65, 15)]
    size = 300
    num_layers = 3
    max_gradient_norm = 1.0
    batch_size = 100
    learning_rate = 1e-4
    use_lstm = False,
    num_samples = 512

    which_cell = "LSTM" if use_lstm else "GRU"
    run_name = str(size) + "unit" + which_cell + "_" + str(num_layers) + "layers"
    checkpoint_directory = "./Runs_50k_20k_vocab/" + run_name + "/"

    what_to_do = input("Train or test?")

    if what_to_do.lower() == "train":
        train_model(checkpoint_directory)

    elif what_to_do.lower() == "test":
        decode("./Runs_50k_20k_vocab/checkpoints/")


