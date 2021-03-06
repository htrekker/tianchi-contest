import argparse


parser = argparse.ArgumentParser()  # 创建解析器，ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息

parser.add_argument(
    "--data_dir",
    # default='data',
    default='data\kaggle',
    type=str,
    # required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--model_type",
    # default="ner",
    default="matching",
    type=str,
    required=False,
    help="Model type selected in the list: "
)
parser.add_argument(
    "--model_name_or_path",
    default='chinese_wwm_pytorch',
    type=str,
    # required=True,
    help="Path to pre-trained model "
)
parser.add_argument(
    "--language",
    default="zh",
    type=str,
    required=False,
    help="Evaluation language. Also train language if `train_language` is set to None.",
)
parser.add_argument(
    "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
)
parser.add_argument(
    "--output_dir",
    default='output',
    type=str,
    # required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--input_test_name",
    # default="train.tsv",
    default="test.tsv",
    type=str,
    # required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--test_ouptut_path",
    default=None,
    type=str,
    # required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)

# Other parameters
parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--task_name",
    # default="weiboner",
    default="lcqmc",
    type=str,
    help="Which task",
)
parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", default=True, action="store_true", help="Whether to run eval on the test set.")
parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
)
parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
)

parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
    "--overwrite_output_dir", default=True, action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


args = parser.parse_args()
