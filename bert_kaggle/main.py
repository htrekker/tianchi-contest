# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import glob
import logging
import os
import random
import sys

# 为import加入搜索目录 ./src/
sys.path.insert(0, "src")
import warnings
# 忽略警告
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.nn import Softmax
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from prepro import processors, output_modes, load_and_cache_examples, compute_metrics, ner_F1
from parser_args import args

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# 分类、匹配、序列标注模型分类
MODEL_CLASSES = {
    "classification": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "matching": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "ner": (BertConfig, BertForTokenClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # 注释：train_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1,  all_labels)
    # 训练的epoch默认3
    args.train_batch_size = args.per_gpu_train_batch_size   # batch default = 32
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:  # default max_steps = -1 "If > 0: set total number of training steps to perform. Override num_train_epochs."
        # 设置num_training_steps=t_total
        t_total = args.max_steps
        # 重新设置epoch
        # // 整数除法
        # default gradient_accumulation_steps = 1
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        # default num_train_epochs = 3.0
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    # 查看是否有训练好的模型
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and \
            os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    set_seed(args)
    # Added here for reproductibility
    for ep in range(int(args.num_train_epochs)):
        print('**********epoch: {}**********'.format(ep+1))
        for step, batch in enumerate(train_dataloader):
            model.train()
            # 数据放入gpu
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            # **将字典中的value当作参数传递给model， *将字典中的key当作参数传递给model
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()

            tr_loss += loss.item()      # .item()转化为标量

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:    # logging_step = 10
                logger.info("global_step = {}, loss = {}".format(global_step, loss.item()))

            if args.save_steps > 0 and global_step % args.save_steps == 0:      # save_step = 500

                results = evaluate(args, model, tokenizer, checkpoint_path='checkpoint-'+str(global_step))
                for key, value in results.items():
                    logger.info("global_step = {}, eval_{} = {}".format(global_step, key, value))
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, checkpoint_path='', prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_examples = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        masks = None
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            m = Softmax(dim=1)
            logits = m(logits)
            if preds is None:
                # detach()切断反向传播 cpu()数据放入cpu上 numpy()将tensor转换成numpy 注意：cuda上的变量只能为tensor，故tensor先放到cpu上再转换成numpy
                preds = logits.detach().cpu().numpy()
                # out_label_ids = inputs["labels"].detach().cpu().numpy()
                masks = inputs["attention_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                masks = np.append(masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)

        kaggle_res = []

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            # preds = np.argmax(preds, axis=1)    # axis=1：按 行 查找最大元素 返回索引
            idxs = np.argmax(preds, axis=1)

            for p, i in zip(preds, idxs):
                # kaggle_res.append(1 / (1 + np.exp(-p[i])))  # sig
                if i == 1:
                    kaggle_res.append(p[i])
                else:
                    kaggle_res.append(1.0 - p[0])
            # result = compute_metrics(preds, out_label_ids)  # 计算acc
        else:
            # ner: logits=scores .shape=[examp_nums, seq_len, num_labels]
            preds = np.argmax(preds, axis=2)
            result = ner_F1(preds, out_label_ids, masks)
            
        # results.update(result)

        output_eval_file = os.path.join(eval_output_dir, checkpoint_path + "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval finish! {} *****".format(prefix))
            # for key in sorted(result.keys()):
            #     logger.info("  %s = %s", key, str(result[key]))
            #     writer.write("%s = %s\n" % (key, str(result[key])))
            for res in kaggle_res:
                writer.write(str(res))
                writer.write('\n')

    return results  # dict {'acc': 0.8123456}


def main():

    if (
        os.path.exists(args.output_dir)
        and any([x for x in os.listdir(args.output_dir) if x.find("bin") > -1])
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args新增n_gpu
    args.n_gpu = 1 if torch.cuda.is_available() else 0
    # args新增device
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()    # 实例化类，processors[args.task_name]为WeiboNerProcessor类
    args.output_mode = output_modes[args.task_name]     # args.output_mode = 'ner'
    label_list = processor.get_labels(args.data_dir)    # 从data目录里找到label.txt并处理返回
    num_labels = len(label_list)

    # Load pretrained model and tokenizer and config
    args.model_type = args.model_type.lower()   # 'bert'
    # 注意顺序
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # BertConfig, BertForTokenClassification, BertTokenizer

    # 加载预训练模型 'chinese_wwm_pytorch'
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,      # matching: num_labels = 2 二分类问题0和1 属于kwargs参数
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # 分词器
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # bert模型 BertForTokenClassification
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # 模型 to gpu
    model.to(args.device)

    # Training
    if args.do_train:
        # args.task_name = 'weiboner'
        train_dataset, train_examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # 训练
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # save model when training finish
    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned (saved at last step)
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # model.to(args.device)
        if args.eval_all_checkpoints:   # 设置为false
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        else:
            checkpoints = [args.output_dir]     # output_dir = 'output'
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()          
