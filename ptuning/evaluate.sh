PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-2e-2
STEP=10
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file /content/ChatGLM2-6B/ptuning/self_data/dev.json \
    --test_file /content/ChatGLM2-6B/ptuning/self_data/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path THUDM/chatglm2-6b \
    --ptuning_checkpoint /content/drive/MyDrive/saved_model/$CHECKPOINT/checkpoint-$STEP \
    --output_dir /content/drive/MyDrive/saved_model/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
