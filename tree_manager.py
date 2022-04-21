import sys
import numpy as np

from vocab import Vocabulary, deserialize_vocab

from nltk.tree import ParentedTree

import DebugFunction as df

class TreeManager:
    
    def __init__(self, vocab):
        
        self.MAX_NB_NODES = 150  # For computational efficiency
        # Selected phrases version 1
        # self.labels_used = [
        #     "NP", "NN", "PP", "VP", "JJ", "VBG", "NNS", "VBZ", "VBN",
        #     "ADVP", "CD", "VBP", "NNP", "ADJP", "VB", "NX",
        #     "VBD", "JJR", "NNPS", "PDT", "RBR"
        # ]
        # Use only phrases related to nouns and verbs
        self.labels_used = [
            "NP", "NN", "VP", "VBG", "NNS", "VBZ", "VBN",
            "VBP", "NNP", "VB", "NX", "VBD", "NNPS"
        ]
        # Use only phrases related to nouns
        # self.labels_used = [
        #     "NP", "NN", "NNS", "NNP", "NX", "NNPS"
        # ]
        
        self.vocab = vocab
        self.tokens = []
        self.index = -1
        self.curr_node = -1 
        self.start_nodes = []
        self.end_nodes = []
        self.node_feats = - np.ones(self.MAX_NB_NODES, dtype=np.int32) 
        # Nodes that will not be used for matching with image regions
        # are marked by "-1"
        self.node_labels = - np.ones(self.MAX_NB_NODES, dtype=np.int32)
        self.sentence = []
        self.tmp_id = -1
        
        
    # Function for traversing a tree
    def traverse(self, t, parent_id):
        try:
            t.label()
        except AttributeError:
            return
        else:
            
            self.curr_node += 1
            my_id = self.curr_node
            # Each node including the root has a label. If the label of the node
            # is included in labels_used or the node is the root, the label ID of
            # the node is set to node_label[my_id]. In other words, node_label for
            # nodes not-used for matching are left as -1. 
            if self.check_used_or_not(t) == True or my_id == 0:
                # print("!! Myself:{} ({}th)".format(t, my_id))
                self.node_labels[my_id] = self.vocab.lab2idx[t.label()]
            # self.node_labels[my_id] = self.vocab.lab2idx[t.label()]  # !!!!!

            # The current node (subtree) is a leaf
            if t.height() == 2:
                # print(
                #     "Myself:{} ({}th) -> Parent:{} ({}th)"
                #     .format(t, my_id, t.parent(), parent_id)
                # )
                self.start_nodes.append(my_id)
                self.end_nodes.append(parent_id)
                # Error check for the ID of this leaf node
                if my_id >= self.MAX_NB_NODES:
                    print(
                        "!!! ERROR ({}th cap): The node id ({}) exceeds the limit ({})"
                        .format(self.index, my_id, self.MAX_NB_NODES)
                    )
                    sys.exit()
                """
                # Error check for the word of this leaf node
                if t[-1].lower() not in self.tokens:
                    print(
                        "!!! WARNING ({}th cap): Unknown token ({} for {})"
                        .format(self.index, t[-1].lower(), self.tokens)
                    )
                    # sys.exit()
                """
                self.node_feats[my_id] = self.vocab(t[-1].lower())
                # print("Label: {} -> {}".format(t.label(), self.node_labels[my_id]))
                # Set the label ID of a non-meaningful node to -1
                if self.check_parent_status(t, t.parent()) == False:
                    self.node_labels[parent_id] = -1
                self.sentence.append(t[-1])
                # if my_id == 27:
                #     df.set_trace()
                return
            
            for child in t:
                self.traverse(child, my_id)
            
            # There is no edge from the root. 
            if parent_id >= 0:
                self.start_nodes.append(my_id)
                self.end_nodes.append(parent_id)
                # Except the root, check the status of the parent node
                if self.check_parent_status(t, t.parent()) == False:
                    self.node_labels[parent_id] = -1
      

    # Function for traversing a tree, used for DEBUGGING
    def traverse_for_debug(self, t, parent_id):
        try:
            t.label()
        except AttributeError:
            return
        else:
            
            self.curr_node += 1
            my_id = self.curr_node
            #df.set_trace()
            if self.node_labels[my_id] >= 0:
                self.tmp_id += 1
                print(
                    "<{}> {}th node -> {} ::: {}"
                    .format(self.tmp_id, my_id, self.node_labels[my_id], t)
                )

            # The current node (subtree) is a leaf
            if t.height() == 2:
                return
            
            for child in t:
                self.traverse_for_debug(child, my_id)
    
    
    def traverse_for_debug2(self, t, target_id):
        try:
            t.label()
        except AttributeError:
            return
        else:

            self.curr_node += 1
            if self.curr_node == target_id:
                return t
            
            # The current node (subtree) is a leaf
            if t.height() == 2:
                return None
            
            for child in t:
                t_ret = self.traverse_for_debug2(child, target_id)
                if t_ret != None:
                    return t_ret
    
    
    def traverse_to_make_phrase_string(self, t):
        try:
            t.label()
        except AttributeError:
            return
        else:
            # The current node (subtree) is a leaf
            if t.height() == 2:
                self.sentence.append(t[-1])
            for child in t:
                self.traverse_to_make_phrase_string(child)
   
    
    def get_phrase_string(self, root, target_id):
        self.curr_node = -1
        target_phrase = self.traverse_for_debug2(root, target_id)
        self.sentence = []
        self.traverse_to_make_phrase_string(target_phrase)
        phrase_string = ""
        for w in self.sentence:
            phrase_string += (w + " ")
        return phrase_string

    
    def check_used_or_not(self, t):
        
        if (t.label() in self.labels_used) == False:
            return False
       
        # A word expressing a number or predeterminer (e.g., all, half etc.)
        # does not make sense by itself.
        if (t.height() == 2) and \
            ( (t.label() == "CD") or (t.label() == "PDT") ):
            return False
        
        return True

    
    # Exclude non-meaningful nodes where two words are contained
    # and one of them is a determiner (e.g., "a" and "the").
    def check_parent_status(self, t, pt):
        
        # The parent is actually the same to the current node
        if len(pt) == 1:
            return False
        
        # Check for the parent of a left node
        # if t.height() == 2:
        #     if (len(pt) == 2) and \
        #        ( (pt[0].label() == "DT") or (pt[1].label() == "DT") ):
        #         return False
        # Check for the parent of a non-leaf node
        # else:
        # Although the parent has two child nodes,
        # their integration is not so meaningful
        if (len(pt) == 2) and \
            ( (pt[0].label() in self.labels_used) == False
            or (pt[1].label() in self.labels_used) == False ):
            # df.set_trace()
            return False
        
        return True

    
    def init(self, tokens, index):
        self.tokens = tokens
        self.index = index
        self.curr_node = -1
        self.start_nodes = []  # .clear()
        self.end_nodes = []  # .clear()
        self.node_feats = - np.ones(self.MAX_NB_NODES, dtype=np.int32)  # -1
        self.node_labels = - np.ones(self.MAX_NB_NODES, dtype=np.int32)
        self.sentence = []
        self.tmp_id = -1
        
        
    def prepare_tree_graph(self, root, tokens, index):
        
        self.init(tokens, index)
        self.traverse(root, self.curr_node)
        #df.set_trace()
        nb_nodes = self.curr_node + 1 
        # print(">> # of nodes = {}".format(nb_nodes))
        # print(">> node_labels: {}".format(self.node_labels[:nb_nodes]))
        
        # Check if there is at least one node used for matching with regions in a frame.
        # If this is not the case, force to use the root.
        if np.sum(self.node_labels >= 0) < 1:
            print(
                "WARNING: No nodes except the root are not used, id={}:{}"
                .format(index, tokens)
            )
            self.node_labels[0] = self.vocab.lab2idx[root.label()]
        
        # For debug
        self.curr_node = -1
        # print("##### {}th: {} #####".format(index, tokens))
        # self.traverse_for_debug(root, self.curr_node)
        #df.set_trace()
        return self.start_nodes, self.end_nodes, self.node_feats[:nb_nodes], self.node_labels[:nb_nodes], self.sentence


# The following main function is used to generate caption files from the
# corresponding tree files. It is also used to generate a file storing
# all the node labels in caption trees.
if __name__ == '__main__':

    trees = []
    
    # This vocab is just for creating TreeManager and will not be used 
    vocab = deserialize_vocab("vocab/coco_precomp_vocab.json")
    print(">> Vocabulary size = {}".format(len(vocab)))
    
    tm = TreeManager(vocab)
    
    tree_filename_train = "data_coco/train_caps_new_trees_20210614.txt"
    with open(tree_filename_train, 'r') as f:
        for i, line in enumerate(f):
            trees.append( ParentedTree.fromstring(line) )
            if i % 100000 == 0:
                print("<train> {}th tree: {}".format(i, trees[-1]))
    print("<train> # of loaded trees = {}".format(len(trees)))
    
    num_trees_train = len(trees)
    
    tree_filename_dev = "data_coco/dev_caps_new_trees_20210615.txt"
    with open(tree_filename_dev, 'r') as f:
        for i, line in enumerate(f):
            trees.append( ParentedTree.fromstring(line) )
            if i % 100000 == 0:
                print("<dev> {}th tree: {}".format(i, trees[-1]))
    print("<dev> # of loaded trees = {}".format(len(trees) - num_trees_train))
    print(">> Total # of loaded trees = {}".format(len(trees)))
    
    
    sentences_from_trees = []
    label_dict = {}

    for i, tree in enumerate(trees):
        
        _, _, _, node_labs, sentence_src = tm.prepare_tree_graph(tree, "", i)
        sentence = ""
        for j in range(len(sentence_src)-1):
            sentence += (sentence_src[j] + " ")
        sentence += sentence_src[-1]
        
        sentences_from_trees.append(sentence)
        for label in node_labs:
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1

        if i % 100000 == 0:
            print("### Processing {}th tree ###".format(i))
            print("-- Node labels: {}".format(node_labs))
            print("-- Sentence: {}".format(sentence))
        # if i == 932:
        #     df.set_trace()
    
    """
    cap_filename_tree = "data_coco/dev_caps_from_trees_20210618.txt"
    print(">> Output sentences into {}".format(cap_filename_tree))
    with open(cap_filename_tree, 'w') as f:
        for s in sentences_from_trees:
            print("{}".format(s), file=f)
    """

    node_labs_filename = "data_coco/node_labels_20210618.txt"
    print(">> Output all the obtained node labels into {}".format(node_labs_filename))
    with open(node_labs_filename, 'w') as f:
        for i, lab in enumerate(label_dict.keys()):
            print("{} {} {}".format(i, lab, label_dict[lab]), file=f)

    # df.set_trace() 
    

