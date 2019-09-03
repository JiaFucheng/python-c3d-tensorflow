
# Steps
1. Put C3D model file *c3d_10000.pb* in *saved_model/* directory.
2. Link UCF-101 dataset (jpg) directory as *dataset/UCF-101*.
3. Run
```shell
# Set GPU device
CUDA_VISIBLE_DEVICES=0 python predict_c3d_ucf101_pb.py
```
