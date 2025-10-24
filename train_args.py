import os
os.system('python train.py \
    --output_dir <output_dir> \
    --dataset_name dataset/drug.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --learning_rate 2e-6 \
    --optim paged_adamw_8bit \
    --do_train \
    --do_eval_on_test\
    --access_token <your_huggingface_token>' )
