"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import time
import itertools
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import dgl

import DebugFunction as df


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        # print("message")
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # print("reduce")
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        # print("apply:{}".format(nodes))
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        print(
            "-- Creating TreeLSTM with num_vocabs:{}, x_size:{}, h_size:{}"
            .format(num_vocabs, x_size, h_size)
        )
        self.x_size = x_size
        self.h_size = h_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        # if pretrained_emb is not None:
        #     print('Using glove')
        #     self.embedding.weight.data.copy_(pretrained_emb)
        #     self.embedding.weight.requires_grad = True
        # self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(h_size, num_classes)
        # cell = TreeLSTMCell if cell_type == 'nary' else ChildSumTreeLSTMCell
        cell = ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)
        
        self.init_embedding_weights()
        self.init_cell_parameters()
        # df.set_trace()
        
        
    def init_embedding_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
    
    def init_cell_parameters(self):
        cell_params = [p for p in list(self.cell.parameters())]
        # print("BEFORE: {}".format(cell_params))
        for p in cell_params:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)
        # print("AFTER: {}".format(cell_params))
        
        
    def forward(self, graph, h_state, c_state):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        graph : dgl.DGLGraph
            Tree for computation.
        h_state : Tensor
            Initial hidden state.
        c_state : Tensor
            Initial cell state.
        Returns
        -------
        h_state : Tensor
            Hidden state after message propagation
        """
        # feed embedding
        row_ids = (graph.ndata["x"] * graph.ndata["mask"]).to(th.int32)
        embeds = self.embedding(row_ids)
        # g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
        # Here, x_j for nodes associated with mask=0 becomes zero
        graph.ndata['iou'] = self.cell.W_iou(embeds) * graph.ndata["mask"].unsqueeze(-1)
        graph.ndata['h'] = h_state
        graph.ndata['c'] = c_state
        # print("h_state (before propagation): {}".format(graph.ndata["h"]))
        # propagate
        dgl.prop_nodes_topo(graph, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # df.set_trace()
        # compute logits
        # h = self.dropout(g.ndata.pop('h'))
        return graph.ndata.pop('h')
        # logits = self.linear(h)
        # return h


