#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /mnt/d/PycharmProjects/models/ChatGLM2-6B \
    --dataset 人群购买_icl,典型理念_icl,理念导入_icl \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0826_all_data_sft \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --quantization_bit 4 \
    --plot_loss \
    --max_source_length 800  \
    --max_target_length 200  \
    --fp16


  #!/bin/bash

#echo "Hello there! pls wait 3 hours"
#sleep 3h
#echo "Oops! I fell asleep for a 3 hours!"
