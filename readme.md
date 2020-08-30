

<!-- ./tools/dist_train.sh configs/faster_rcnn_r101_fpn_1x.py 8 --validate --eval_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x/syn_feat.pth -->

<!-- python tools/zero_shot_utils.py  configs/faster_rcnn_r101_fpn_1x.py --extract_feats_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x/syn_feat.pth -->
<!-- extract feats -->
<!-- python tools/zero_shot_utils.py  configs/faster_rcnn_r101_fpn_1x.py --extract_feats_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth  -->

./tools/dist_train.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py 2 --validate --eval_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth --syn_weights /raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/VOC/classifier_best.pth

./tools/dist_train.sh configs/faster_rcnn_r101_fpn_1x.py 1 --validate --eval_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth --syn_weights /raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/coco/classifier_best.pth

python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --extract_feats_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth

./tools/dist_train.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py 8 --validate --eval_only --resume_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/syn_feats.pth










