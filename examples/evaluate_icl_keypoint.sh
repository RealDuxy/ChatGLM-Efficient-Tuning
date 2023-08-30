#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
  --stage sft \
  --model_name_or_path  /mnt/d/PycharmProjects/models/ChatGLM2-6B \
  --do_predict  \
  --dataset_dir ../data \
  --dataset icl_eval_keypoint  \
  --finetuning_type lora  \
  --checkpoint_dir  ../checkpoints/0-content_0803 \
  --output_dir  ../outputs/0-content_0803-keypoint_grdy \
  --per_device_eval_batch_size  4 \
  --predict_with_generate \
  --max_source_length 2048  \
  --max_target_length 500  \

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
  --stage sft \
  --model_name_or_path  /mnt/d/PycharmProjects/models/ChatGLM2-6B \
  --do_predict  \
  --dataset_dir ../data \
  --dataset icl_eval_keypoint  \
  --finetuning_type lora  \
  --checkpoint_dir  ../checkpoints/3-content_0803 \
  --output_dir  ../outputs/3-content_0803-keypoint_grdy \
  --per_device_eval_batch_size  4 \
  --predict_with_generate \
  --max_source_length 2048  \
  --max_target_length 500  \

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
  --stage sft \
  --model_name_or_path  /mnt/d/PycharmProjects/models/ChatGLM2-6B \
  --do_predict  \
  --dataset_dir ../data \
  --dataset icl_eval_keypoint  \
  --finetuning_type lora  \
  --checkpoint_dir  ../checkpoints/0-keypoint_0803 \
  --output_dir  ../outputs/0-keypoint_0803-keypoint_grdy \
  --per_device_eval_batch_size  4 \
  --predict_with_generate \
  --max_source_length 2048  \
  --max_target_length 500  \

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
  --stage sft \
  --model_name_or_path  /mnt/d/PycharmProjects/models/ChatGLM2-6B \
  --do_predict  \
  --dataset_dir ../data \
  --dataset icl_eval_keypoint  \
  --finetuning_type lora  \
  --checkpoint_dir  ../checkpoints/2-keypoint_0803 \
  --output_dir  ../outputs/2-keypoint_0803-keypoint_grdy \
  --per_device_eval_batch_size  4 \
  --predict_with_generate \
  --max_source_length 2048  \
  --max_target_length 500  \

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
  --stage sft \
  --model_name_or_path  /mnt/d/PycharmProjects/models/ChatGLM2-6B \
  --do_predict  \
  --dataset_dir ../data \
  --dataset icl_eval_keypoint  \
  --finetuning_type lora  \
  --checkpoint_dir  ../checkpoints/3-keypoint_0803 \
  --output_dir  ../outputs/3-keypoint_0803-keypoint_grdy \
  --per_device_eval_batch_size  4 \
  --predict_with_generate \
  --max_source_length 2048  \
  --max_target_length 500  \

