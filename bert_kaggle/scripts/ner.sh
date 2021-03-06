# 指定GPU
export CUDA_VISIBLE_DEVICES=2
train_name=$1
model_dir=../../bert_pytorch/chinese_wwm_pytorch/
# data_dir=data/chnsenticorp
# data_dir=data/lcqmc
data_dir=data/weiboner


if [ ! "$train_name" ]; then
    echo "modelname is none"
    exit 1
fi

if [ ! -d "$train_name" ]; then
    mkdir $train_name
fi

python main.py \
    --model_type=ner \
    --task_name=weiboner \
    --data_dir=$data_dir \
    --model_name_or_path=$model_dir \
    --output_dir=$train_name \
    --max_seq_length=30 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=256 \
    --learning_rate=5e-5 \
    --weight_decay=0.0 \
    --warmup_steps=0 \
    --save_steps=100 \
    --logging_steps=10 \
    --num_train_epochs=3 \
    --overwrite_output_dir \
    --evaluate_during_training
