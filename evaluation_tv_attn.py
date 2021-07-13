import sys
import os
import numpy as np
import re

from vocab import Vocabulary, deserialize_vocab
import torch
from model import SCAN, func_attention

from nltk.tree import ParentedTree
from tree_manager import TreeManager
import dgl

import cv2

import DebugFunction as df


def xattn_score_t2i_check(images, captions, cap_lens, opt):
    
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_cap_len = np.max(cap_lens)
    attn_list = []  # s = np.zeros((n_image, n_caption, max_cap_len))
    
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        _, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        attn_list.append(attn)
    
    return attn_list
    

def make_topic_batch(topic_tree_path, vocab, target_topic_id):
    
    print(">> Load topic trees from {}".format(topic_tree_path))
    topic_trees = []
    with open(topic_tree_path, 'r') as f:
        for line in f:
            topic_trees.append( ParentedTree.fromstring(line) )
    print(">> # of loaded topic trees = {}".format(len(topic_trees)))
    
    tree_manager = TreeManager(vocab)
    
    topic_graphs = []
    # topic_graphs_node_nums = []
    topic_graphs_node_labs = []
    target_topic_phrases = []
    for i, topic_tree in enumerate(topic_trees):

        # It is OK to feed an empty string as tokens of prepare_tree_graph
        start_nodes, end_nodes, node_feats, node_labels, _ = tree_manager.prepare_tree_graph(topic_tree, [""], i)
        
        # Nodes whose features are -1 are masked by 0, and
        # the other nodes are assocaited with 1 indicating they are used.
        node_masks = (node_feats != -1).astype(np.float32)
        topic_graph = dgl.graph(
            (torch.tensor(start_nodes, dtype=torch.int32), torch.tensor(end_nodes, dtype=torch.int32))
        )
        topic_graph.ndata["x"] = torch.from_numpy(node_feats)
        topic_graph.ndata["mask"] = torch.from_numpy(node_masks)
        # topic_graph.ndata["y"] = torch.from_numpy(node_labels)
        # print("id:{}\nx:{}".format(index, target.ndata["x"]))
        # print("x_org: {} ({})".format(node_feats, len(node_feats)))
        # print("mask:{}\ny:{}".format(target.ndata["mask"], target.ndata["y"]))
        # print("start_nodes:{}".format(start_nodes))
        # print("end_nodes:{}".format(end_nodes))
        topic_graphs.append(topic_graph)
        # topic_graphs_node_nums.append(topic_graph.num_nodes())
        topic_graphs_node_labs.append(node_labels)
    
        if i == target_topic_id:
            for j, node_label in enumerate(node_labels):
                if node_label >= 0:
                    phrase_string = tree_manager.get_phrase_string(topic_trees[target_topic_id], j)
                    print("- {} - {}".format(j, phrase_string))
                    target_topic_phrases.append(phrase_string)
    # df.set_trace()

    return dgl.batch(topic_graphs), topic_graphs_node_labs, target_topic_phrases
    

def main(model_cp_path, topic_tree_path, region_dir, target_shot_id, target_topic_id):
    
    # load model and options
    print(">> Load a model's checkpoint from {}".format(model_cp_path))
    checkpoint = torch.load(model_cp_path)
    opt = checkpoint['opt']
    print(opt)
    
    # load vocabulary used by the model
    full_vocab_path = os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)
    print(">> Load the vocabulary from {}".format(full_vocab_path))
    vocab = deserialize_vocab(full_vocab_path, True)
    assert len(vocab) == opt.vocab_size, \
        "Vocabulary size mismatch ({} vs. {})".format(len(vocab), opt.vocab_size)
    
    # construct model
    model = SCAN(opt)
    
    # load model state
    model.load_state_dict(checkpoint['model'])
    
    # Make a batch collecting all the topics' trees 
    topic_graphs, topic_graphs_node_labs, target_topic_phrases = make_topic_batch(topic_tree_path, vocab, target_topic_id)
    # Compute embeddings of nodes in each of topic trees
    topic_nodes_emb, topic_node_nums = model.forward_txt_emb(
        topic_graphs, topic_graphs_node_labs, True
    )
    
    topic_num = len(topic_graphs_node_labs)
    # List where each element is a list containing similarities of all shots to a certain topic
    # df.set_trace()
    
    videoID_str = (region_dir.split("/"))[-2]
    print("### Processing shots in {}th video ###".format(videoID_str))
    npy_filename = "data_tv/npyfile_precomp/" + videoID_str + ".npy"
    print(">> Load shot features from {}".format(npy_filename))
    shot_feats = np.load(npy_filename)
    shot_num = shot_feats.shape[0]
    print(">> # of shots = {}".format(shot_num))
    
    shot_feats_emb = model.forward_img_emb(torch.Tensor(shot_feats), True)
    
    attns = xattn_score_t2i_check(shot_feats_emb, topic_nodes_emb, topic_node_nums, opt)
    # df.set_trace()
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 128),
        (0, 255, 128), (128, 255, 0), (128, 0, 255), (0, 128, 255),
    ]
    color_num = len(colors)
    print(
        "##### Target topic ID:{}, target shot ID:{} #####"
        .format(target_topic_id, target_shot_id)
    )
    target_shot_keyframe_filename = "../TRECVID2021/keyframe/" + videoID_str \
                                    + "/shot" + videoID_str + "_" + str(target_shot_id) + "_RKF.png"
    # NOTE: In each element of attns, the attentions between regions in result of each shot
    # and words in a topic is stored at the position whose ID starts with 0.
    target_attn = attns[target_topic_id][target_shot_id - 1].cpu().numpy()
    print(
        ">> Draw the most attentive region for each of {} nodes on the keyframe {}"
        .format(target_attn.shape[1], target_shot_keyframe_filename)
    )
    keyframe = cv2.imread(target_shot_keyframe_filename)
    
    region_filename = region_dir + "shot" + videoID_str + "_" + str(target_shot_id) + "_RKF.png.box.npy"
    print(">> Load regions from {}".format(region_filename))
    regions = np.load(region_filename)
    
    attn_region_ids = np.argmax(target_attn, 0)
    for i, attn_region_id in enumerate(attn_region_ids):
        
        keyframe_copy = keyframe.copy()
        attn_region = (np.around(regions[attn_region_id])).astype(int)
        print(
            "--- Most attentive region for {}th phrase ({}) => {}"
            .format(i, target_topic_phrases[i], attn_region)
        )
        cv2.rectangle(keyframe_copy,
            (attn_region[0], attn_region[1]),
            (attn_region[2], attn_region[3]),
            colors[i % color_num], thickness=3
        )
        txt_bottom_left = (
            min(attn_region[0], attn_region[2]),
            min(attn_region[1], attn_region[3]) - 10
        )
        cv2.putText(
            keyframe_copy, "{}: {}".format(i, target_topic_phrases[i]), txt_bottom_left,
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i % color_num], thickness=3
        )
        # df.set_trace()
    
        output_keyframe_filename = region_dir  + "shot" + videoID_str + "_" + str(target_shot_id) + "_RKF_attn_t" + str(i) + ".png"
        print(">> Save the keyframe with the attentive region into {}".format(output_keyframe_filename))
        cv2.imwrite(output_keyframe_filename, keyframe_copy)
    
    
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python evaluation_tv.py <Path to a model's checkpoint>"
          + " <Path to a topic tree file> <Directory of region information>"
          + " <Target shot ID (starting from 1)> <Target topic ID (3 digits)>"
          + " <Topic ID offset (611 for tv19, 641 for tv20)>\n"
          + "NOTE: Attentions for 36 regions in each frame (shot) are stored"
          + " in the directory of region information."
        )
        sys.exit()
    main(
        sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]), int(sys.argv[5]) - int(sys.argv[6])
    )

