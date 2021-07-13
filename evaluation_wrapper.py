import sys

from vocab import Vocabulary
import evaluation

model_path = sys.argv[1]
data_path = sys.argv[2]
print(">> model_path = {}".format(model_path))
print(">> data_path = {}".format(data_path))

evaluation.evalrank(model_path, data_path=data_path, split="test")

