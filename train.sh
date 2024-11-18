CUDA_VISIBLE_DEVICES=7 torchrun --master_port 7811 --nproc_per_node=1 train.py \
  --depth=12 --bs=16 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path /data1/feng/data/ImageNet