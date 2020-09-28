

# This is implementstion of Synthesising the unseen for zero shot object detection.

Zero Shot Detection (ZSD) is a recently introduced paradigm which enables simultaneous localization and classification of previously unseen objects. It is arguably the most extreme case of learning with minimal supervision. we propose a symantically driven conditional feature generation module to synthesize visual features for unseen objects. 

<!-- ### Feature Generation Pipeline -->
![](images/pipeline.png)


### Test 
    MSCOCO

        ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best_137.pth

    PASCALVOC
        
        ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 8 --dataset voc --out voc_results.pkl  --zsd --syn_weights ../checkpoints/VOC/classifier_latest.pth

    
        ./tools/dist_test.sh configs/ilsvrc/faster_rcnn_r101_fpn_1x.py work_dirs/ILSVRC/epoch_12.pth 6 --out ilsrvc_results.pkl --dataset imagenet --zsd --syn_weights ../checkpoints/imagenet_0.6_1_0_1_w2v/classifier_best_
