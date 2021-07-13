This SCAN matches regions in an image with phrases/words in a query by extracting textual features of those phrases/words with Tree LSTM.

As pre-processing to run this code, the constituency tree needs to be extracted for each caption or query using the following neural parser:

https://github.com/princeton-vl/attach-juxtapose-parser

In addition to libraries necessary for SCAN, this code needs the following library:

https://docs.dgl.ai/

This libaray is used to make a batch of constituency trees as a big graph, and perform batch computation to propagate hidden states of each of nodes (i.e., phrases/words).
