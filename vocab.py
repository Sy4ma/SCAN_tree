# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Vocabulary wrapper"""

import nltk
from collections import Counter
import argparse
import os
import json

import DebugFunction as df

# annotations = {
#     'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
#     'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
# }
annotations = {
    # "coco_precomp": ["train_caps_new_20210614.txt", "dev_caps_new_20210615.txt"]        
    "coco_precomp": ["train_caps_from_trees_20210618.txt", "dev_caps_from_trees_20210618.txt"]        
}

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, with_phrase_labs=False):
        
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        if with_phrase_labs == True:
            self.lab2idx = {}
            self.idx2lab = {}
            phrase_label_filename = "data_coco/node_labels.txt"
            print(">> Load phrase labels from {}".format(phrase_label_filename))
            with open(phrase_label_filename) as f:
                for line in f:
                    line = line[:-1]
                    line_elems = line.split(" ")
                    # print("lab:{}, id:{}".format(line_elems[1], line_elems[0]))
                    self.lab2idx[line_elems[1]] = int(line_elems[0])
                    self.idx2lab[int(line_elems[0])] = line_elems[1]
            print(">> # of phrase labels = {}".format(len(self.lab2idx)))
        # df.set_trace()

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src, with_phrase_labs=False):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary(with_phrase_labs)
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, caption_file, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for path in caption_file[data_name]:
        full_path = os.path.join(data_path, path)
        print(">> Load words from {}".format(full_path))
        captions = from_txt(full_path)
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # df.set_trace()

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=4)
    # df.set_trace()
    serialize_vocab(vocab, './vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", './vocab/%s_vocab.json' % data_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data_coco')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f30k}_precomp')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
