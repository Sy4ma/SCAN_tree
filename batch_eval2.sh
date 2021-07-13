#!/bin/sh

python -u evaluation_tv.py runs/coco_scan_nounverb_related_relaxed/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2019_611-640.txt runs_tv/2019/nounverb_related_relaxed/

python -u evaluation_tv.py runs/coco_scan_nounverb_related_relaxed/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2020_641-660.txt runs_tv/2020/nounverb_related_relaxed/
