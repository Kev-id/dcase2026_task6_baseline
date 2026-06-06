import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from span_utils import generalized_temporal_iou, span_cxw_to_xx
from matcher import build_matcher
from qd_detr_transformer import build_transformer
from position_encoding import build_position_encoding
from misc import accuracy


def inverse_sigmoid(x, eps=1e-3):
    #这个函数负责计算输入张量的逆sigmoid值。作用是将sigmoid函数的输出值转换回原始输入值。它首先将输入张量的值限制在0和1之间，然后计算逆sigmoid值并返回结果。
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class QDDETR(nn.Module):
    #这个类定义了QD-DETR模型。它继承自nn.Module，并在初始化方法中接受多个参数来配置模型的结构和行为。模型包含一个变压器模块、位置编码模块、文本位置编码模块、多个线性层用于输入投影，以及一个多层感知机用于预测时间跨度和类别。前向方法接受文本和音频输入，并通过变压器进行处理，最终输出预测的时间跨度和类别，以及一些辅助信息用于损失计算。
    def __init__(
        self,
        transformer,
        position_embed,
        txt_position_embed,
        aud_dim,
        txt_dim,
        num_queries,
        input_dropout, 
        max_a_l,
        aux_loss=True,
        span_loss_type="l1", 
        use_txt_pos=False,
        n_input_proj=2
    ):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single audio.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_a_l: int, maximum #clips in audio
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_a_l = max_a_l
        span_pred_dim = 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False

        self.input_txt_proj = nn.Sequential(*[
            #这些线性层用于将文本输入投影到隐藏维度空间。它们可以配置为使用层归一化、dropout和ReLU激活函数。根据n_input_proj参数的值，可能会使用不同数量的线性层进行投影。
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_aud_proj = nn.Sequential(*[
            #这些线性层用于将音频输入投影到隐藏维度空间。它们可以配置为使用层归一化、dropout和ReLU激活函数。根据n_input_proj参数的值，可能会使用不同数量的线性层进行投影。此外，第一个线性层还将位置嵌入添加到音频输入中，以提供位置信息。
            LinearLayer(aud_dim + 2, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]), # add pos_embedding
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.aux_loss = aux_loss#这个布尔参数指示是否使用辅助解码损失。如果为True，模型将在每个解码器层的输出上计算损失，而不仅仅是在最后一层。这可以帮助模型在训练过程中更好地学习，并可能提高性能。

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))


    def forward(self, src_txt, src_txt_mask, src_aud, src_aud_mask):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_aud: [batch_size, L_aud, D_aud]
               - src_aud_mask: [batch_size, L_aud], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_aud = self.input_aud_proj(src_aud)#这些线性层用于将音频输入投影到隐藏维度空间。
        src_txt = self.input_txt_proj(src_txt)#这些线性层用于将文本输入投影到隐藏维度空间。
        src = torch.cat([src_aud, src_txt], dim=1)  # (bsz, L_aud+L_txt, d)。为什么要cat？因为变压器的输入是一个序列，所以我们需要将音频和文本输入连接在一起形成一个长序列。
        mask = torch.cat([src_aud_mask, src_txt_mask], dim=1).bool()  # (bsz, L_aud+L_txt)
        pos_aud = self.position_embed(src_aud, src_aud_mask)  # (bsz, L_aud, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_aud, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_aud+L_txt, d)

        # for global token
        #这段代码负责为变压器输入添加一个全局表示的token。作用是提供一个全局的上下文表示，帮助模型更好地理解整个输入序列。首先，它创建一个与输入序列长度相同的掩码，并将全局表示token的掩码设置为True。然后，它将全局表示token添加到输入序列的开头，并将相应的位置编码添加到位置编码中。最后，它将更新后的输入序列、掩码和位置编码传递给变压器进行处理。
        #这个token之后用于从变压器的输出中提取全局表示，并与音频片段的表示进行交互，以计算saliency分数和对比损失。
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        audio_length = src_aud.shape[1]
        
        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, audio_length)
        #变压器的输出包括解码器的隐藏状态hs、参考点reference、变压器编码器的输出memory以及全局表示memory_global。hs的形状为(#layers, batch_size, #queries, hidden_dim)，reference的形状为(batch_size, #queries, 2)，memory的形状为(batch_size, L_aud+L_txt+1, hidden_dim)，memory_global的形状为(batch_size, hidden_dim)。这些输出将用于后续的预测和损失计算。
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)#这个线性层用于将解码器的隐藏状态hs映射到类别预测空间。它接受形状为(#layers, batch_size, #queries, hidden_dim)的输入，并输出形状为(#layers, batch_size, #queries, 2)的张量，表示每个查询的类别预测（背景或前景）。输出中的最后一个维度大小为2，因为我们有两个类别：背景和前景。
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)#这个多层感知机用于预测时间跨度。它接受解码器的隐藏状态hs作为输入，并输出一个形状为(#layers, batch_size, #queries, 2)的张量，表示每个查询的预测时间跨度（中心位置和宽度）。根据span_loss_type参数的值，输出可能会经过sigmoid函数进行归一化，以确保预测值在0和1之间。
        outputs_coord = tmp + reference_before_sigmoid#这个操作将多层感知机的输出与参考点进行相加，以获得最终的预测时间跨度。参考点是变压器解码器中每个查询的初始位置估计，通过添加多层感知机的输出，模型可以调整这个位置以更准确地预测事件的时间跨度。如果span_loss_type是l1，那么输出将经过sigmoid函数进行归一化，以确保预测值在0和1之间，表示相对于输入序列长度的比例。
        if self.span_loss_type == "l1":#如果span_loss_type参数的值是"l1"，则表示使用L1损失来计算时间跨度的损失。在这种情况下，输出将经过sigmoid函数进行归一化，以确保预测值在0和1之间，表示相对于输入序列长度的比例。这是因为L1损失通常用于回归任务，而时间跨度的预测需要在一个连续的范围内进行，因此需要将输出限制在合理的范围内。
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}#一个是类别预测，另一个是时间跨度预测。outputs_class[-1]表示使用最后一层解码器的输出进行类别预测，而outputs_coord[-1]表示使用最后一层解码器的输出进行时间跨度预测。这些预测将用于计算损失和生成最终的提交结果。
        #下面的代码负责从变压器编码器的输出中提取文本和音频的表示，并计算saliency分数和对比损失。首先，它将变压器编码器的输出memory分割为文本表示txt_mem和音频表示aud_mem。然后，它构建负样本对，通过将文本输入进行循环移位来创建负样本，并将其与音频输入连接在一起形成负样本输入。接下来，它将正样本输入和负样本输入分别传递给变压器，得到对应的编码器输出memory_global和memory_global_neg。最后，它使用这些输出计算saliency分数和对比损失，并将它们添加到输出字典中。
        txt_mem = memory[:, src_aud.shape[1]:]  # (bsz, L_txt, d)
        aud_mem = memory[:, :src_aud.shape[1]]  # (bsz, L_aud, d)
            
        ### Neg Pairs ###
        #通过将文本输入进行循环移位来创建负样本，并将其与音频输入连接在一起形成负样本输入。这种方法可以生成与正样本不同的文本输入，从而帮助模型学习区分正负样本，提高模型的鲁棒性和泛化能力。
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_aud, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_aud_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()  # since it does not use actual content

        _, _, memory_neg, memory_global_neg = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, audio_length)
        aud_mem_neg = memory_neg[:, :src_aud.shape[1]]

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(aud_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(aud_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["audio_mask"] = src_aud_mask
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        eos_coef,
        losses, 
        span_loss_type,
        max_a_l,
        saliency_margin=1
    ):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_a_l = max_a_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        aud_token_mask = outputs["audio_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * aud_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        aud_token_mask = aud_token_mask.repeat([1, 2])
        saliency_scores = aud_token_mask * saliency_scores + (1. - aud_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * aud_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": loss_saliency}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        indices = self.matcher(outputs_without_aux, targets)
        losses_target = self.losses

        # Compute all the requested losses
        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                losses_target = self.losses

                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = QDDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        max_a_l=args.max_a_l,
        txt_dim=args.t_feat_dim,
        aud_dim=args.a_feat_dim,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
    )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_saliency": args.lw_saliency
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        span_loss_type=args.span_loss_type, 
        max_a_l=args.max_a_l,
        saliency_margin=args.saliency_margin,
    )
    criterion.to(device)
    return model, criterion
