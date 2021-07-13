import sys
import os
import numpy as np
import re

from vocab import Vocabulary, deserialize_vocab
import torch
from model import SCAN, xattn_score_t2i

from nltk.tree import ParentedTree
from tree_manager import TreeManager
import dgl

import DebugFunction as df


# Slightly customised version of shard_xattn_t2i in evaluation.py
def shard_xattn_t2i_tv(images, captions, caplens, opt, shard_size=512):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int( (len(images)-1)/shard_size + 1 )
    n_cap_shard = int( (len(captions)-1)/shard_size + 1 )
    # print(
    #     "n_im_shard: {}, n_cap_shard: {}".format(n_im_shard, n_cap_shard)
    # )
    d = np.zeros((len(images), len(captions)))
    # df.set_trace()
    
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):

            with torch.no_grad():
                sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
                im = images[im_start:im_end]
                s = captions[cap_start:cap_end]
                l = caplens[cap_start:cap_end]
                sim = xattn_score_t2i(im, s, l, opt)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
            # df.set_trace()

    sys.stdout.write('\n')
    return d


def make_topic_batch(topic_tree_path, vocab):
    
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
    # df.set_trace()
    
    return dgl.batch(topic_graphs), topic_graphs_node_labs
    

def main(model_cp_path, shot_npy_dir, topic_tree_path, output_dir):
    
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
    topic_graphs, topic_graphs_node_labs = make_topic_batch(topic_tree_path, vocab)
    # Compute embeddings of nodes in each of topic trees
    topic_nodes_emb, topic_node_nums = model.forward_txt_emb(
        topic_graphs, topic_graphs_node_labs, True
    )
    # df.set_trace()
    
    topic_num = len(topic_graphs_node_labs)
    # List where each element is a list containing similarities of all shots to a certain topic
    sims_all = [[] for i in range(topic_num)]
    
    NUM_VIDEOS = 7475
    shot_infos_all = []
    for i in range(1, NUM_VIDEOS+1):
       
        # if i != 3934:
        #     continue
        print("### Processing shots in {}th video ###".format(i))
        
        npy_filename = shot_npy_dir + "{:05d}.npy".format(i)
        print(">> Load shot features from {}".format(npy_filename))
        shot_feats = np.load(npy_filename)
        print(">> # of shots = {}".format(shot_feats.shape[0]))
        
        shot_feats_emb = model.forward_img_emb(torch.Tensor(shot_feats), True)
        
        sims = shard_xattn_t2i_tv(
            shot_feats_emb, topic_nodes_emb, topic_node_nums, opt
        )
        # df.set_trace()
        for j in range(topic_num):
            sims_all[j] += sims[:,j].tolist()
        
        video_id = "{:05d}".format(i)
        for j in range(1, shot_feats.shape[0]+1):
            shot_infos_all.append( video_id + "_" + str(j) )
        # if i == 20:
        #     break  # df.set_trace()
    
    print(">> Total number of shots = {}".format(len(shot_infos_all)))

    sims_all_np = np.stack(sims_all, axis=0)
    sims_filename = output_dir + "sims_all.npy"
    print("### Output similarities of all shots to each topic into {} ###".format(sims_filename))
    np.save(sims_filename, sims_all_np)
    
    print("### Output retrieval results under {} ###".format(output_dir))
    topic_id_start = int( re.split("[_.-]", topic_tree_path)[-3] )
    for i in range(topic_num):
        
        topic_id = i + topic_id_start
        output_filename = output_dir + str(topic_id) + ".trec"
        print(">> Output {}th topic result into {}".format(topic_id, output_filename))
        
        # Sort shots based on their similarities to the topic_id-th topic
        ranks_one_topic = np.argsort( np.array(sims_all[i]) )
        ranks_one_topic = ranks_one_topic[::-1]
        
        with open(output_filename, 'w') as fout:
            for j in range(1000):
                if j == 0 or j== 999:
                    print(
                        "--- {}th ranked shot: {} (sim:{})"
                        .format(j, shot_infos_all[ranks_one_topic[j]], sims_all[i][ranks_one_topic[j]])
                    )
                print(
                    "{} 0 shot{} {} {} trec"
                    .format(1000+topic_id, shot_infos_all[ranks_one_topic[j]], j+1, 9999-j),
                    file=fout
                )
        
        # if i == 1:
        #     df.set_trace()
    
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python evaluation_tv.py <Path to a model's checkpoint> \
                <Directory of shot npy files> <Path to a topic tree file> <Output directory>"
        )
        sys.exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

