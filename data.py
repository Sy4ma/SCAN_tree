# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod

from nltk.tree import ParentedTree
from tree_manager import TreeManager
import dgl

import DebugFunction as df


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        
        print("### Creating PrecompDataset for {} ###".format(data_split))
        self.vocab = vocab
        loc = data_path + '/'
        full_path_cap = loc + data_split + "_caps.txt"
        print(">> Load captions from {}".format(full_path_cap))
            
        # Captions
        self.captions = []
        with open(full_path_cap, 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        print(">> # of loaded captions = {}".format(len(self.captions)))
        
        full_path_tree = loc + data_split + "_caps_trees.txt"
        print(">> Load caption trees from {}".format(full_path_tree))
        
        # Trees
        self.trees = []
        with open(full_path_tree, 'r') as f:
            for line in f:
                self.trees.append( ParentedTree.fromstring(line)  )
        print(">> # of loaded trees = {}".format(len(self.trees)))
        
        self.tree_manager = TreeManager(vocab)
        # df.set_trace()
        
        # Image features
        full_path_img = loc + data_split + "_ims.npy"
        print(">> Load visual features from {}".format(full_path_img))
        self.images = np.load(full_path_img)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev':
        #     self.length = 5000
        # if data_split == "dev":
        #     df.set_trace()

    def __getitem__(self, index):
        # index = 0 
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab 
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        # print("** index = {}: {}".format(index, tokens))
        start_nodes, end_nodes, node_feats, node_labels, _ = self.tree_manager.prepare_tree_graph(self.trees[index], tokens, index)
        #cap = self.tree_manager.get_phrase_string(self.trees[0], 0)
        # Nodes whose features are -1 are masked by 0, and
        # the other nodes are assocaited with 1 indicating they are used. 
        # df.set_trace()
        node_masks = (node_feats != -1).astype(np.float32)
        target = dgl.graph(
            (torch.tensor(start_nodes, dtype=torch.int32), torch.tensor(end_nodes, dtype=torch.int32))
        )
        target.ndata["x"] = torch.from_numpy(node_feats)
        target.ndata["mask"] = torch.from_numpy(node_masks) 
        # target.ndata["y"] = torch.from_numpy(node_labels) 
        # print("id:{}\nx:{}".format(index, target.ndata["x"]))
        # print("x_org: {} ({})".format(node_feats, len(node_feats)))
        # print("mask:{}\ny:{}".format(target.ndata["mask"], target.ndata["y"]))
        # print("start_nodes:{}".format(start_nodes))
        # print("end_nodes:{}".format(end_nodes))
        
        # caption = []
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # target = torch.Tensor(caption)
        # print("Org: {} ({})".format(self.captions[index], len(self.captions[index].split(" "))))
        # for i, cid in enumerate(caption):
        #     print("{}th word: {} -> {}".format(i, cid, vocab.idx2word[str(cid)]))
        # df.set_trace()
        
        #return image, target, node_labels, index, img_id, start_nodes, end_nodes
        return image, target, node_labels, start_nodes, end_nodes, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[1]), reverse=True)
    #images, cap_trees, node_labs, ids, img_ids, st_nodes, end_nodes = zip(*data)
    images, cap_trees, node_labs, st_nodes, end_nodes, ids, img_ids = zip(*data)
    #df.set_trace()
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    #  i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]
    
    # Make a batch of caption trees
    targets = dgl.batch(cap_trees)
    # This lengths seems to be not needed, but currently anyway kept.\
    # lengths = [cap_tree.num_nodes() for cap_tree in cap_trees]
    # df.set_trace()
    
    return images, targets, node_labs, st_nodes, end_nodes, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)
    #df.set_trace()
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    #df.set_trace()
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = opt.data_path  # os.path.join(opt.data_path, data_name)
    print(">> Create a data loader from {}".format(dpath))
    # train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
    #                                   batch_size, True, workers)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = opt.data_path  # os.path.join(opt.data_path, data_name)
    print(">> Create a data loader for TEST from {}".format(dpath))
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
