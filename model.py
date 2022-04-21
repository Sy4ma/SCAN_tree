# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import OrderedDict

import sys

from tree_lstm import TreeLSTM
from tree_manager import TreeManager

import DebugFunction as df


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', 
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    print(
        ">> Initialising image encoder for {} (type:{})"
        .format(data_name, precomp_enc_type)
    )
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        print(
            "++ Creating an image encoder with img_dim:{}, embed_size:{}"
            .format(img_dim, embed_size)
        )
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)
        
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        print(">> Initialising text encoder")
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()
        # df.set_trace()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        
        # Forward propagate RNN
        out, _ = self.rnn(packed)
        
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        
        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:int(cap_emb.size(2)/2)] + cap_emb[:,:,int(cap_emb.size(2)/2):])/2
        
        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        
        return cap_emb, cap_len


def func_attention(query, context, attn_prev, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)
    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # df.set_trace()
    
    weight_factor = 0.7
    attnT = weight_factor * attnT + (1.0 - weight_factor) * attn_prev
    #attnT = attnT + attn_prev
    
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # df.set_trace()
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    
    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    #return (w12 / (w1 * w2).clamp(min=eps))
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def find_leafnode(st_node, ed_node):
    """ find leaf node"""
    leaves = st_node.copy()
    for element in ed_node:
        try:
            leaves.remove(element)
        except ValueError:
            pass
    #l_poss = list(range(len(leaves)))
    return leaves

def find_nextend(st_node, next_st, ed_node):
    next_ed = []
    for element in next_st:
        next_ed.append(ed_node[st_node.index(element)])

    return next_ed

def xattn_score_t2i(images, captions, cap_lens, st_nodes, end_nodes, opt, is_shard=False):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    #leaves = []
    #for i in range(len(st_nodes)):
    #    leaves.append(list(set(st_nodes[i]) - set(end_nodes[i])))

    #st = st_nodes[0]
    #ed = end_nodes[0]
    #leaf = find_leafnode(st, ed)
    #next_st = find_leafnode(st, leaf)
    #next_ed = find_nextend(st, next_st, ed)
    """
    for i in range(1):
        print("--------- {} tree node ----------".format(i))
        st = st_nodes[i]
        ed = end_nodes[i]
        st_prev = st
        ed_prev = ed
        print("start",st)
        print("end", ed)
        while True:
            leaves = find_leafnode(st_prev, ed_prev)
            print("leaves",leaves)
            next_st = find_leafnode(st_prev, leaves)
            if len(next_st) == 0:
                break
            next_ed = find_nextend(st, next_st, ed)
            print("Next_start", next_st)
            print("Next_end", next_ed)
            st_prev = next_st
            ed_prev = next_ed
    """
    # df.set_trace()
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    #print("n_images: ",images.shape)
    #print("n_captions: ", captions.shape)
    for n in range(n_caption):
        
        zero_flag = False
        st = st_nodes[n]
        ed = end_nodes[n]
        st_prev = None
        ed_prev = None
        n_word = cap_lens[n]
        cap_i = captions[n, :n_word, :].unsqueeze(0).contiguous()
        nodes = sorted(list(set(st) | set(ed)))
        batch_size = images.shape[0]
        areas = images.shape[1]
        node_features = torch.zeros(batch_size, areas, len(nodes)).to("cuda")
        weiContext_all = torch.zeros(batch_size, len(nodes), images.shape[2]).to("cuda")

        children = {e: [] for e in ed}
        for j, e in enumerate(ed):
            if e == 0 and st[j] == 0:
                continue
            children[e].append(st[j])
        # df.set_trace()
        
        tmp_cnt = 0
        while True:
            # array of ids indicating the current leaf nodes
            leaves = find_leafnode(st, ed)  # l_poss
            # pick up features of the current leaf nodes from cap_i -> cap_i2
            cap_i2 = captions[n, leaves, :].unsqueeze(0).contiguous() 
            # expand cap_i2
            cap_i2_expand = cap_i2.repeat(n_image, 1, 1)
            cap_i2_expand = cap_i2_expand.contiguous()
            # df.set_trace()
            # attn_prev2 is created from attn_prev using l_poss
            # att_prev2 is defined as a dictionary key:curr_ed. value: list of features
            # for curr_ed in ed:
            #   for prev_st in st_prev:
            #       if curr_ed == prev_st:
            #          # check if curr_ed exists in attn_prev2
                       # if not, create a new entry
                       # if exist, add feature to value (list of feature
            print("st", st)
            print("ed", ed)
            print("leaves: ", leaves)
            print("children: ", children)
            attn_prev2 = {leaf: [] for leaf in leaves}
            for leaf in leaves:
                if leaf not in children:
                    continue
                for child in children[leaf]:
                    index = nodes.index(child)
                    if torch.sum(node_features[:,:,index]) == 0:
                        continue
                    attn_prev2[leaf].append(node_features[:,:,index])
            # create attn_prev3 with 128 x 36 x (# of entries in attn_prev2)
            #print("-----------------------------------------------")
            #df.set_trace()
            #print(attn_prev2)
            # attn_prev3.shape([128,36,# of leaves])
            attn_prev3 = torch.zeros(batch_size, areas, len(attn_prev2)).to("cuda")
            # check each entry of attn_prev2
            for i, k in enumerate(attn_prev2.keys()):
                for feature in attn_prev2[k]:
                    #df.set_trace()
                    attn_prev3[:,:,i] += feature
                #df.set_trace()
                attn_prev3[:,:,i] /= max(len(attn_prev2[k]), 1)
            # if tmp_cnt == 1:
            #     df.set_trace()
            
            # compute the average feature for the list of features of the current entry
            weiContext, attn = func_attention(cap_i2_expand, images, attn_prev3, opt, smooth=opt.lambda_softmax)
            # if is_shard == True:
            #     df.set_trace()
            weiContext = weiContext.contiguous()
            weiContext_all[:,leaves,:] = weiContext.to(torch.float32)
            # Shallow copy? Deep copy?
            for i, leaf in enumerate(leaves):
                index = nodes.index(leaf)
                node_features[:,:,index] = attn[:,:,i]
            next_st = find_leafnode(st, leaves)
            next_ed = find_nextend(st, next_st, ed)
            if len(next_st) == 0:
                if zero_flag:
                    break
                next_st = [0]
                next_ed = []
                zero_flag = True
            #df.set_trace()
            st_prev = st
            ed_prev = ed
            st = next_st
            ed = next_ed
            tmp_cnt += 1
        # Get the i-th text description
        #n_word = cap_lens[i]
        #cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        #cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        #weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        #cap_i_expand = cap_i_expand.contiguous()
        #weiContext = weiContext.contiguous()
        # (n_image, n_word)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        row_sim = cosine_similarity(cap_i_expand, weiContext_all, dim=2)
        #df.set_trace()
        # Note if n_image == 1 and n_word == 1, it is needed to perform unsqueeze twice.
        if row_sim.dim() <= 1:
            if n_image == 1:
                row_sim = row_sim.unsqueeze(0)
            if n_word == 1:
                row_sim = row_sim.unsqueeze(1)       
        
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        
        #df.set_trace()
        similarities.append(row_sim)
    
    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    # df.set_trace()
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)
    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        print("** Creating contractive loss (margin:{})".format(margin))
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l, st, end):
        # compute image-sentence score matrix
        # df.set_trace() 
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, st, end, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        #print("------------------------scores-------------------------")
        #print(scores)
        #print(torch.isnan(scores).any())
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        
        # df.set_trace()
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        #df.set_trace()
        # self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
        #                         opt.embed_size, opt.num_layers, 
        #                         use_bi_gru=opt.bi_gru,  
        #                         no_txtnorm=opt.no_txtnorm)
        self.txt_enc = TreeLSTM(opt.vocab_size, opt.word_dim, opt.embed_size)
        #df.set_trace() 
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
        
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        #df.set_trace()
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        
        self.params = params
        print("------------------------------------------------------------------------------------")
        print(params) 
        print("------------------------------------------------------------------------------------")
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        
        self.Eiters = 0
        #df.set_trace()
        
    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict
        
    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        
    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        
    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
   
    
    def organise_caption_embeddings(self, cap_emb, node_labs):
        # Arrange cap_emb so that its shape is batch_size x max node number x embedding size
        cap_num = len(node_labs)
        node_nums = []
        used_node_nums = []
        for c_node_labs in node_labs:
            node_nums.append(c_node_labs.size)
            used_node_num = np.sum(c_node_labs >= 0)
            used_node_nums.append(used_node_num)
        max_used_node_num = max(used_node_nums)
        
        if torch.cuda.is_available():
            cap_emb2 = torch.zeros(cap_num, max(node_nums), self.txt_enc.h_size).to("cuda")
            #cap_emb2 = torch.zeros(cap_num, max_used_node_num, self.txt_enc.h_size).to("cuda")
        else:
            cap_emb2 = torch.zeros(cap_num, max(node_nums), self.txt_enc.h_size)
            #cap_emb2 = torch.zeros(cap_num, max_used_node_num, self.txt_enc.h_size)
        
        # !!! Change cap_emb2 so that it contains all node features
        #cap_emb2 = torch.reshape(cap_emb, (cap_num, max(node_nums), self.txt_enc.h_size))
               
        row_offset = 0
        for cid in range(cap_num):
            # print("cid:{} -> {}-{}:".format(cid, row_offset, row_offset+node_nums[cid]))
            # cap_emb_poss = np.nonzero(node_labs[cid] >= 0)[0] + row_offset
            # print("{}th rows are copied".format(cap_emb_poss))
            cap_emb2[cid,:node_nums[cid],:] = cap_emb[row_offset:row_offset+node_nums[cid],:]
            row_offset += node_nums[cid]
        print(
            "--- cap_emb ({}) is transformed into cap_emb2 ({})"
            .format(cap_emb.shape, cap_emb2.shape)
        )
        # df.set_trace()

        return cap_emb2, node_nums
    
    
    def forward_emb(self, images, cap_trees, node_labs, volatile=False):
        """Compute the image and caption embeddings
        """
        if volatile == False:
            # Set mini-batch dataset
            images = Variable(images)  # , volatile=volatile)
            # cap_trees = Variable(cap_trees) # , volatile=volatile)
            if torch.cuda.is_available():
                
                images = images.cuda()
                
                cap_trees = cap_trees.to("cuda")
                node_num = cap_trees.number_of_nodes()
                h_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                c_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                # cap_trees = cap_trees.cuda()
                #df.set_trace() 
            # Forward
            img_emb = self.img_enc(images)
            
            # cap_emb (tensor), cap_lens (list)
            cap_emb = self.txt_enc(cap_trees, h_state, c_state)
            cap_emb2, node_nums2 = self.organise_caption_embeddings(cap_emb, node_labs)
            
            return img_emb, cap_emb2, node_nums2

        else:
            with torch.no_grad():
                
                images = Variable(images)
                if torch.cuda.is_available():
                    images = images.cuda()
                    cap_trees = cap_trees.to("cuda")
                    node_num = cap_trees.number_of_nodes()
                    h_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                    c_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                
                img_emb = self.img_enc(images)
                
                # cap_emb (tensor), cap_lens (list)
                cap_emb = self.txt_enc(cap_trees, h_state, c_state)
                cap_emb2, node_nums2 = self.organise_caption_embeddings(cap_emb, node_labs)
                # df.set_trace()
                
                return img_emb, cap_emb2, node_nums2
         
        
    def forward_img_emb(self, images, volatile=False):
        if volatile == False:
            images = Variable(images)  # , volatile=volatile)
            if torch.cuda.is_available():
                images = images.cuda()
            return self.img_enc(images)
        else:
            with torch.no_grad():
                images = Variable(images)
                if torch.cuda.is_available():
                    images = images.cuda()
                return self.img_enc(images)
    
    
    def forward_txt_emb(self, cap_trees, node_labs, volatile=False):
        if volatile == False:
            if torch.cuda.is_available():
                cap_trees = cap_trees.to("cuda")
                node_num = cap_trees.number_of_nodes()
                h_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                c_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
            cap_emb = self.txt_enc(cap_trees, h_state, c_state)
            return self.organise_caption_embeddings(cap_emb, node_labs)
        else:
            with torch.no_grad():
                if torch.cuda.is_available():
                    cap_trees = cap_trees.to("cuda")
                    node_num = cap_trees.number_of_nodes()
                    h_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                    c_state = torch.zeros((node_num, self.txt_enc.h_size)).to("cuda")
                cap_emb = self.txt_enc(cap_trees, h_state, c_state)
                return self.organise_caption_embeddings(cap_emb, node_labs)
    
    
    def forward_loss(self, img_emb, cap_emb, cap_len, st_nodes, end_nodes, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        #df.set_trace()
        loss = self.criterion(img_emb, cap_emb, cap_len,st_nodes, end_nodes)
        self.logger.update('Le', loss.data.item(), img_emb.size(0))
        return loss
    
    def train_emb(self, images, cap_trees, node_labs, st_nodes, end_nodes, ids=None, *args):
        """One training step given images and captions.
           NOTE: cap_trees is a batch of multiple caption trees, and
                 this batch is aggregated into one graph.
        """
        #ids = None
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        #df.set_trace() 
        # compute the embeddings
        img_emb, cap_emb, node_nums = self.forward_emb(images, cap_trees, node_labs)
        # df.set_trace()  
        # measure accuracy and record loss
        
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, node_nums, st_nodes, end_nodes)
        #print("--------------------------------loss-----------------------------")
        #print(loss)
        #print(torch.isnan(loss))
        # df.set_trace()
        # compute gradient and do SGD step
        loss.backward()
        
        print("------------------------------------------------------------------") 
        #print(self.params)
        print("----nan----")
        print(any(torch.isnan(i).any().item() for i in self.params))
        print("----inf----")
        print(any(torch.isinf(i).any().item() for i in self.params))
        print("------------------------------------------------------------------")
        
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        

