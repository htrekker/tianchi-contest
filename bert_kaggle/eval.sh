export CUDA_VISIBLE_DEVICES=2
train_name=$1
model_dir=../../bert_pytorch/chinese_wwm_pytorch/
data_dir=data/chnsenticorp


if [ ! "$train_name" ]; then
    echo "modelname is none"
    exit 1
fi

if [ ! -d "$train_name" ]; then
    mkdir $train_name
fi

python main.py \
    --model_type=classification \
    --task_name=chnsenti \
    --data_dir=$data_dir \
    --model_name_or_path=$model_dir \
    --output_dir=$train_name \
    --max_seq_length=128 \
    --do_eval \
    --per_gpu_eval_batch_size=256 \
    --overwrite_output_dir \
    --evaluate_during_training
