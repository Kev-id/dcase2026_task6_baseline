import os
import time
import json
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from easydict import EasyDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from evaluate import eval_epoch, start_inference, setup_model

from basic_utils import AverageMeter, dict_to_markdown, write_log, save_checkpoint, rename_latest_to_best
from model_utils import count_parameters, ModelEMA

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(
        model,
        criterion, 
        train_loader, 
        optimizer, 
        opt, 
        epoch_i
    ):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()#criterion是什么？criterion是一个损失函数，用于计算模型输出与目标之间的差距。在训练过程中，模型会根据输入数据生成预测结果，然后使用criterion来计算这些预测结果与实际标签之间的损失值。这个损失值会被用来指导模型的参数更新，以使模型在训练过程中逐渐提高性能。常见的损失函数包括均方误差（MSE）、交叉熵损失（CrossEntropyLoss）等，具体使用哪种损失函数取决于任务的类型和需求。

    # init meters
    loss_meters = defaultdict(AverageMeter)#loss_meters是一个字典，使用了defaultdict和AverageMeter类来跟踪和计算不同类型的损失值的平均值。每当你访问一个新的键时，defaultdict会自动创建一个新的AverageMeter实例，这样你就可以方便地更新和计算每种损失的平均值，而不需要担心键不存在的问题。这对于训练过程中监控多个损失指标非常有用，可以帮助你更好地了解模型的性能和训练进展。

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),#tqdm是什么库？它是一个Python库，用于在命令行界面中显示进度条。它可以帮助开发者更直观地了解代码的执行进度，特别是在处理大量数据或长时间运行的任务时非常有用。
                                 desc="Training Iteration",
                                 total=num_training_examples):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)

        outputs = model(**model_inputs, targets=targets) if opt.model_name == 'cg_detr' else model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:#grad_clip是什么？grad_clip是一个超参数，用于控制梯度的最大值，以防止在训练过程中出现梯度爆炸的问题。当模型的梯度过大时，可能会导致模型参数更新过大，从而使训练过程不稳定。通过设置grad_clip的值，可以将梯度的范数限制在一个指定的范围内，从而保持训练过程的稳定性。常见的做法是使用torch.nn.utils.clip_grad_norm_函数来实现梯度裁剪。
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    write_log(opt, epoch_i, loss_meters)


def train(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_dataset, 
        val_dataset, 
        opt
    ):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,#collate_fn是什么？collate_fn是PyTorch DataLoader中的一个参数，用于指定如何将一个批次的数据样本组合成一个批次的张量。当你使用DataLoader加载数据时，它会从数据集中取出多个样本，并将它们组合成一个批次。默认情况下，DataLoader会使用一个简单的函数来组合这些样本，但有时候你可能需要自定义这个过程，比如处理不同长度的序列、进行数据增强等。这时，你可以定义一个自己的collate_fn函数，并将其传递给DataLoader，以便在加载数据时使用你的自定义逻辑来组合样本。
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):#trange约等于range。desc="Epoch"的意思是在进度条前面显示"Epoch"，total=num_training_examples的意思是设置进度条的总长度为num_training_examples，这样进度条就能正确地显示训练的进度。
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()#lr_scheduler是什么？lr_scheduler是一个学习率调度器，用于在训练过程中动态调整学习率。它可以根据预设的策略（如每隔一定的epoch数降低学习率，或者根据验证集的性能调整学习率）来改变优化器的学习率，从而帮助模型更好地收敛。常见的学习率调度器包括StepLR、ExponentialLR、ReduceLROnPlateau等，具体使用哪种调度器取决于你的训练需求和模型表现。

        if opt.model_ema:#model_ema是什么？model_ema是一个模型的指数移动平均（Exponential Moving Average）版本。在训练过程中，model_ema会根据当前模型的参数更新自己的参数，以便在评估和推理阶段使用更稳定的模型参数。通过使用model_ema，可以在训练过程中获得更好的性能和泛化能力，因为它能够平滑模型参数的更新，减少过拟合的风险。在评估阶段，你可以选择使用model_ema的参数来进行评估，以获得更准确的性能指标。
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:#这段的意思是每隔opt.eval_epoch_interval个epoch进行一次评估。例如，如果opt.eval_epoch_interval设置为5，那么在第5、10、15等epoch结束时会执行评估代码块。这种做法可以帮助你在训练过程中定期检查模型的性能，而不需要在每个epoch结束时都进行评估，从而节省时间和计算资源。
            with torch.no_grad():
                if opt.model_ema:
                    metrics, eval_loss_meters, latest_file_paths = \
                        eval_epoch(model_ema.module, val_dataset, opt, save_submission_filename, criterion)
                else:
                    metrics, eval_loss_meters, latest_file_paths = \
                        eval_epoch(model, val_dataset, opt, save_submission_filename, criterion)

            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val')            
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            
            stop_score = metrics["brief"]["MR-full-R1@0.7"]

            if stop_score > prev_best_score:
                prev_best_score = stop_score
                save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                logger.info("The checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)


def main(opt, resume=None):#resume是一个可选参数，表示是否从一个已经存在的模型检查点继续训练。如果resume参数不为None，说明用户希望从指定的检查点文件中加载模型的状态，以便继续之前的训练过程。这通常用于在训练过程中断后恢复训练，或者在进行微调时使用预训练模型的权重。通过加载检查点，模型可以从之前的训练状态继续学习，而不需要从头开始训练，从而节省时间和计算资源。
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)

    # dataset & data loader
    dataset_config = EasyDict(#EasyDict是什么？EasyDict是一个Python库，提供了一种方便的方式来创建和访问字典对象。与普通的字典不同，EasyDict允许你使用点（.）来访问字典中的键，而不是使用方括号（[]）。例如，如果你有一个EasyDict对象config，你可以通过config.key来访问其中的值，而不需要使用config['key']。这使得代码更加简洁和易读，特别是在处理配置参数时非常有用。
        data_path=opt.train_path,
        ctx_mode=opt.ctx_mode,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        a_feat_type=opt.a_feat_type,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )

    train_dataset = StartEndDataset(**dataset_config)  #StartEndDataset是一个自定义的数据集类，通常用于处理具有起始和结束位置的任务，例如文本分类、序列标注等。在这个类中，你可以定义如何加载数据、预处理数据以及如何返回模型所需的输入格式。通过继承PyTorch的Dataset类，StartEndDataset可以与DataLoader配合使用，以便在训练过程中批量加载数据并进行迭代。具体实现细节取决于你的任务需求和数据结构，但总体来说，StartEndDataset旨在为模型提供适当格式化的数据输入，以便进行训练和评估。
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.val_path
    eval_dataset = StartEndDataset(**copied_eval_config)
    
    # prepare model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)

    logger.info(f"Model {model}")
    count_parameters(model, verbose=True)

    if resume is not None:
        checkpoint = torch.load(resume, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded model checkpoint: {}".format(resume))

    logger.info("Start Training...")
    
    # start training
    train(
        model,
        criterion,
        optimizer,
        lr_scheduler, 
        train_dataset, 
        eval_dataset, 
        opt
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config path')
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="specify model path for fine-tuning. If None, train the model from scratch.",
    )
    args = parser.parse_args()
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option
    main(opt, args.resume)
