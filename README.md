# ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth 8 --syn_weights /raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/coco/classifier_best.pth --out coco_results.pkl
# ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py cp_48_17/epoch_12.pth 8 --syn_weights /raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/coco_48_17/classifier_latest.pth --out coco_results.pkl
# ./tools/dist_test.sh configs/ilsvrc/faster_rcnn_r101_fpn_1x.py work_dirs/ILSVRC/epoch_12.pth 6 --out ilsrvc_results.pkl --syn_weights /raid/mun/codes/zero_shot_detection/zsd_ilsvrc/checkpoints/imagenet_0.6_1_0_1_exp_5/classifier_best_70.pth
# ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 8 --dataset voc --out voc_results.pkl --syn_weights /raid/mun/codes/zero_shot_detection/zsd_abl/checkpoints/VOC/classifier_best_36.pth

# ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth 8 --out coco_results.pkl --dataset coco --syn_weights /raid/mun/codes/zero_shot_detection/zsd_coco/checkpoints/coco_65_15_0.6_1_0_1/classifier_best_137.pth

<!-- extract feats VOC -->
# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes seen --load_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712_contrastive/epoch_4.pth --save_dir ../../data/voc_con --data_split train
<!-- COCO -->
# python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth --save_dir ../../data/coco --data_split train_default


# python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes unseen --load_from work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth --save_dir ../../data/coco --data_split test_default


# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes unseen --load_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712_contrastive/epoch_4.pth --save_dir ../../data/voc_con --data_split test

# ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712_contrastive/epoch_4.pth 8 --dataset voc --out voc_results.pkl --syn_weights /raid/mun/codes/zero_shot_detection/zsd_abl/checkpoints/VOC_con/classifier_latest_104.pth --zsd

# ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights /raid/mun/codes/zero_shot_detection/zsd_abl/checkpoints/coco_65_15_default/classifier_best_15.pth 

<!-- work dirs -->
coco_con_with_noise_l2
faster_rcnn_r101_fpn_1x
coco_65_15_default



### Test 
    MSCOCO

        ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best_137.pth

    PASCALVOC
        
        ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 8 --dataset voc --out voc_results.pkl  --zsd --syn_weights ../checkpoints/VOC/classifier_latest.pth

    
        ./tools/dist_test.sh configs/ilsvrc/faster_rcnn_r101_fpn_1x.py work_dirs/ILSVRC/epoch_12.pth 6 --out ilsrvc_results.pkl --dataset imagenet --zsd --syn_weights ../checkpoints/imagenet_0.6_1_0_1_w2v/classifier_best_