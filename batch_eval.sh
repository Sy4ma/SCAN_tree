#!/bin/sh

echo "python -u evaluation_tv.py runs/coco_scan_all_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2021_661-680.txt runs_tv/2021/all_phrases/"
python -u evaluation_tv.py runs/coco_scan_all_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2021_661-680.txt runs_tv/2021/all_phrases/


echo "python -u evaluation_tv.py runs/coco_scan_all_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_p_591-610.txt runs_tv/2021_p/all_phrases/"
python -u evaluation_tv.py runs/coco_scan_all_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_p_591-610.txt runs_tv/2021_p/all_phrases/


echo "python -u ../SCAN2/evaluation_tv.py runs/coco_scan_selected_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2021_661-680.txt runs_tv/2021/selected_phrases/"
python -u ../SCAN2/evaluation_tv.py runs/coco_scan_selected_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2021_661-680.txt runs_tv/2021/selected_phrases/


echo "python -u ../SCAN2/evaluation_tv.py runs/coco_scan_selected_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_p_591-610.txt runs_tv/2021_p/selected_phrases/"
python -u ../SCAN2/evaluation_tv.py runs/coco_scan_selected_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_p_591-610.txt runs_tv/2021_p/selected_phrases/


echo "python -u ../SCAN3/evaluation_tv.py runs/coco_scan_noun_verb_related_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2021_661-680.txt runs_tv/2021/noun_verb_related_phrases/"
python -u ../SCAN3/evaluation_tv.py runs/coco_scan_noun_verb_related_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_2021_661-680.txt runs_tv/2021/noun_verb_related_phrases/


echo "python -u ../SCAN3/evaluation_tv.py runs/coco_scan_noun_verb_related_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_p_591-610.txt runs_tv/2021_p/noun_verb_related_phrases/"
python -u ../SCAN3/evaluation_tv.py runs/coco_scan_noun_verb_related_phrases/model/model_best.pth.tar data_tv/npyfile_precomp/ data_tv/trees_topics_p_591-610.txt runs_tv/2021_p/noun_verb_related_phrases/



