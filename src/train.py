import os
import time
import json
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm
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
    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)

        outputs = model(**model_inputs, targets=targets) if opt.model_name == 'cg_detr' else model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
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
        opt,
        start_epoch=0,
        prev_best_score=0,
    ):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    epoch_iterator = tqdm(
        range(start_epoch, opt.n_epoch),
        desc="Epoch",
        total=opt.n_epoch,
        initial=start_epoch,
    )
    latest_checkpoint_path = os.path.join(opt.results_dir, "latest_checkpoint.pth")

    for epoch_i in epoch_iterator:
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if opt.model_ema:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
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
                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_i,
                    opt,
                    best_score=prev_best_score,
                )
                logger.info("The best checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)

        # Save an exact training-resume point after every completed epoch. This
        # is separate from the best validation checkpoint used for fine-tuning.
        save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            epoch_i,
            opt,
            checkpoint_path=latest_checkpoint_path,
            best_score=prev_best_score,
        )
        logger.info("Saved latest training checkpoint: %s", latest_checkpoint_path)


def main(opt, resume=None, resume_training=None):
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)
    # Experiment configs commonly use a new output directory. The released
    # training code otherwise fails only after finishing the first epoch.
    os.makedirs(opt.results_dir, exist_ok=True)

    # dataset & data loader
    dataset_config = EasyDict(
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

    train_dataset = StartEndDataset(**dataset_config)    
    copied_eval_config = copy.deepcopy(dataset_config)
    # The released configs expose the validation annotation as ``val_path``.
    # ``eval_path`` is not defined in either config.yml or the pretraining config.
    copied_eval_config.data_path = opt.val_path
    eval_dataset = StartEndDataset(**copied_eval_config)
    
    # prepare model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)

    logger.info(f"Model {model}")
    count_parameters(model, verbose=True)

    start_epoch = 0
    prev_best_score = 0

    if resume_training is not None:
        checkpoint = torch.load(resume_training, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        prev_best_score = checkpoint.get("best_score") or 0
        logger.info(
            "Resuming training from %s at epoch %d with best score %.4f",
            resume_training,
            start_epoch,
            prev_best_score,
        )
    elif resume is not None:
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
        opt,
        start_epoch=start_epoch,
        prev_best_score=prev_best_score,
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
    parser.add_argument(
        "--resume-training",
        type=str,
        help=(
            "resume an interrupted run, including optimizer, scheduler, epoch, "
            "and best validation score"
        ),
    )
    args = parser.parse_args()
    if args.resume is not None and args.resume_training is not None:
        parser.error("--resume and --resume-training cannot be used together")
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option
    main(opt, args.resume, args.resume_training)
