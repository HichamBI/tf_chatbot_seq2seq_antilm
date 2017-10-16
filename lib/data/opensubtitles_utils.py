import sys, os
import tensorflow as tf
from tensorflow.python.platform import gfile

from lib import data_utils

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = data_utils.initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)

                    utterences = line.split('\t')

                    tokenized_utterences = []
                    for utter in utterences:
                        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(utter), vocab,
                                                          tokenizer, normalize_digits)
                        tokenized_utterences.append(" ".join([str(tok) for tok in token_ids]))

                    tokens_file.write("\t".join(tokenized_utterences) + "\n")


def prepare_opensubtitles_data(data_dir, vocabulary_size):
    train_path = os.path.join(data_dir, 'opensubtitles.train')
    dev_path = os.path.join(data_dir, 'opensubtitles.dev')

    vocab_path = os.path.join(data_dir, "vocab%d.in" % vocabulary_size)
    data_utils.create_vocabulary(vocab_path, train_path + ".in", vocabulary_size)

    # Create token ids for the training data.
    train_ids_path = train_path + (".ids%d.in" % vocabulary_size)
    data_to_token_ids(train_path + ".in", train_ids_path, vocab_path)

    # Create token ids for the development data.
    dev_ids_path = dev_path + (".ids%d.in" % vocabulary_size)
    data_to_token_ids(dev_path + ".in", dev_ids_path, vocab_path)

    return (train_ids_path, dev_ids_path, vocab_path)


def read_data(tokenized_dialog_path, buckets, max_size=None, reversed=False):
  """Read data from source file and put into buckets.

  Args:
    source_path: path to the files with token-ids.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in buckets]

  with gfile.GFile(tokenized_dialog_path, mode="r") as fh:
      utterences = fh.readline().split('\t')
      source = utterences[0] if len(utterences) >= 2 else None
      target = utterences[1] if len(utterences) >= 2 else None

      if reversed:
        source, target = target, source  # reverse Q-A pair, for bi-direction model
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

        utterences = fh.readline().split('\t')
        source = utterences[0] if len(utterences) >= 2 else None
        target = utterences[1] if len(utterences) >= 2 else None
  return data_set
