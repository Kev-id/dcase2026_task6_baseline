import argparse
import pprint

from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from easydict import EasyDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basic_utils import AverageMeter
from span_utils import span_cxw_to_xx

from config import BaseOptions

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission

from basic_utils import save_jsonl, save_json
from qd_detr import build_model as build_model_qd_detr

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    #这个函数负责在评估阶段进行后处理。它首先将提交结果保存为JSONL文件，然后根据评估分割的名称（val或test）决定是否进行评估。如果是val或test分割，它调用eval_submission函数来评估提交结果，并将评估指标保存为JSON文件。最后，它返回评估指标和最新的文件路径列表。
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val", "test"]:
        metrics = eval_submission(submission, gt_data)
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    return metrics, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, criterion=None):
    #这个函数负责计算模型在评估数据上的结果。它首先根据模型名称选择适当的批处理输入函数，然后初始化一个字典来存储损失的平均值。接下来，它遍历评估数据加载器中的每个批次，获取查询元数据、模型输入和目标，并将输入传递给模型以获得输出。然后，它从输出中提取预测的时间跨度和概率，并根据概率对预测进行排序。最后，它使用PostProcessorDETR对结果进行后处理，并返回最终的结果和损失平均值。
    batch_input_fn = cg_detr_prepare_batch_inputs if opt.model_name == 'cg_detr' else prepare_batch_inputs
    loss_meters = defaultdict(AverageMeter)

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = batch_input_fn(batch[1], opt.device)
        outputs = model(**model_inputs)

        # compose predictions
        pred_spans = outputs["pred_spans"].cpu()  # (bsz, #queries, 2)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        scores = prob[..., 0].cpu()  # * (batch_size, #queries)  foreground label is 0, we directly take it

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):            
            spans = span_cxw_to_xx(spans) * meta["duration"]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
            )

            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length, min_ts_val=0, max_ts_val=300,
        min_w_l=1, max_w_l=300, move_window_method="left",
        process_func_names=("clip_ts", "round_multiple")
    )

    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, criterion):
    #这个函数负责获取评估结果。它调用compute_mr_results函数来计算模型在评估数据上的结果和损失平均值，并返回这些结果。
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, criterion)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion):
    #这个函数负责在评估阶段执行一个评估周期。它首先设置模型为评估模式，并创建一个数据加载器来加载评估数据集。然后，它调用get_eval_res函数来获取评估结果和损失平均值。接下来，它调用eval_epoch_post_processing函数来对结果进行后处理，并返回评估指标、损失平均值和最新的文件路径列表。
    logger.info("Generate submissions")
    model.eval()
    criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, criterion)        
    metrics, latest_file_paths = eval_epoch_post_processing(
        submission, opt, eval_dataset.data, save_submission_filename)
    return metrics, eval_loss_meters, latest_file_paths


def setup_model(opt):
    #这个函数负责设置模型、优化器和学习率调度器，并在需要时加载检查点。它首先调用build_model_qd_detr函数来构建模型和损失函数，然后将它们移动到指定的设备（如果是CUDA）。接下来，它创建一个AdamW优化器，使用模型中所有需要梯度更新的参数，并设置学习率和权重衰减。最后，它创建一个StepLR学习率调度器，根据指定的学习率下降步数来调整学习率。函数返回模型、损失函数、优化器和学习率调度器。
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model_qd_detr(opt)

    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    return model, criterion, optimizer, lr_scheduler


def start_inference(opt):
    #这个函数负责开始推理过程。它首先记录一条日志，表示正在设置配置、数据和模型。然后，它根据评估分割的名称（val或test）设置数据集配置，并创建一个StartEndDataset对象来加载评估数据集。接下来，它调用setup_model函数来设置模型、损失函数、优化器和学习率调度器，并加载指定的模型检查点。最后，它记录一条日志，表示开始推理，并调用eval_epoch函数来执行评估周期，并记录评估指标。
    logger.info("Setup config, data and model...")

    # dataset & data loader
    dataset_config = EasyDict(
        data_path=opt.val_path if opt.eval_split_name == 'val' else opt.test_path,
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
    
    eval_dataset = StartEndDataset(**dataset_config)
    model, criterion, _, _ = setup_model(opt)
    checkpoint = torch.load(opt.model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    logger.info("Model checkpoint: {}".format(opt.model_path))

    logger.info("Starting inference...")
    save_submission_filename = "submission.jsonl"

    with torch.no_grad():
        metrics, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion)
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics["brief"], indent=4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='config path')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--split', '-s', type=str, default='val', choices=['val', 'test'], help='split name: val or test')
    args = parser.parse_args()#这里使用argparse库来解析命令行参数，要求用户提供一个配置文件路径（--config）和一个模型检查点路径（--model_path）。用户还可以选择评估的分割（--split），默认为'val'，也可以选择'test'。
    option_manager = BaseOptions(args.config)#这行创建了一个BaseOptions对象，传入用户提供的配置文件路径。BaseOptions类负责读取和解析配置文件中的参数，并将其存储在一个属性中。
    option_manager.parse()
    opt = option_manager.option#这行调用了option_manager的parse方法，解析配置文件中的参数，并将结果存储在opt属性中。opt现在包含了从配置文件中读取的所有参数，可以通过opt.attribute_name来访问这些参数。

    opt.model_path = args.model_path
    opt.eval_split_name = args.split
    start_inference(opt)
