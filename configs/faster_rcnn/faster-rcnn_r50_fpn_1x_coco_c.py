_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1))
    
    )

# 修改数据集相关配置
dataset_type = 'CocoDataset'
data_root = '/home/ubuntu/mmdetection/datasets/luderick_base/'
metainfo = {
    'classes': ('luderick', ),

}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/ubuntu/mmdetection/datasets/luderick_base/annotations/train.json',
        data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/ubuntu/mmdetection/datasets/luderick_base/annotations/val.json',
        data_prefix=dict(img='images/val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file='/home/ubuntu/mmdetection/datasets/luderick_base/annotations/val.json')
test_evaluator = val_evaluator

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = '/home/ubuntu/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'