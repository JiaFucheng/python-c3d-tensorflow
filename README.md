
# Steps
1. Put C3D model file in **saved_model/** directory.
2. Link UCF-101 jpeg dataset directory in **dataset/**.
3. Run
```shell
CUDA_VISIBLE_DEVICES=0 python predict_c3d_ucf101_pb.py
```
