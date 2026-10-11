"""Compare fixed-checkpoint B2/M1 fine-tuning seeds on CASTELLA.

Run after analyze_baseline_by_duration.py for each method, seed and split.
Error bars show the range of two fine-tuning seeds, not confidence intervals.
"""
from pathlib import Path
import csv
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'b2_m1_comparison'
SEEDS = (2023, 2024)
METHODS = ('b2', 'm1')
KEYS = ('r1_at_0.5_percent', 'r1_at_0.7_percent',
        'width_error_median', 'overwide_percent', 'abs_center_error_median')


def load_summary(method, seed, split):
    path = RESULTS / f'{method}_seed{seed}_{split}_by_duration' / 'summary.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        return {r['duration_bin']: r for r in csv.DictReader(f)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    tables = {}
    for split in ('val', 'test'):
        for seed in SEEDS:
            b = load_summary('b2', seed, split)
            m = load_summary('m1', seed, split)
            query_keys = []
            for method in METHODS:
                p = RESULTS / f'{method}_seed{seed}_{split}_by_duration' / 'per_query.csv'
                with p.open(encoding='utf-8-sig', newline='') as f:
                    query_keys.append([(r['duration_bin'], r['qid']) for r in csv.DictReader(f)])
            assert query_keys[0] == query_keys[1], 'Query groups must align across models'
            for label in b:
                assert b[label]['queries'] == m[label]['queries']
                for key in KEYS:
                    bv, mv = float(b[label][key]), float(m[label][key])
                    rows.append(dict(split=split, seed=seed, duration_bin=label,
                                     metric=key, b2=bv, m1=mv, delta=mv-bv))
            for method, values in (('b2', b), ('m1', m)):
                tables[split, seed, method] = values
    short = [r for r in rows if r['duration_bin'] == '<5 s' and r['metric'] in KEYS[:2]]
    assert len(short) == 8 and all(r['delta'] < 0 for r in short), 'Recheck report conclusions'
    with (OUT / 'paired_deltas.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ['# B2–M1 配对实验分析', '',
             '每种方法使用一个固定预训练 checkpoint，再分别以 2023、2024 两个种子微调。'
             'checkpoint 仅按验证集 R1@0.7 选择；测试集用于最终报告。', '',
             '## 测试集总体结果', '',
             '| seed | 方法 | R1@0.5 | R1@0.7 | mAP |',
             '|---|---|---:|---:|---:|']
    overall = {}
    for seed in SEEDS:
        for method in METHODS:
            path = RESULTS / f'{method}_seed{seed}_evaluation' / 'test_metrics.json'
            metrics = json.loads(path.read_text(encoding='utf-8'))['brief']
            overall[method, seed] = metrics
            vals = [metrics[k] for k in ('MR-full-R1@0.5', 'MR-full-R1@0.7', 'MR-full-mAP')]
            lines.append(f'| {seed} | {method.upper()} | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    for method in METHODS:
        vals = [statistics.mean(overall[method, s][k] for s in SEEDS)
                for k in ('MR-full-R1@0.5', 'MR-full-R1@0.7', 'MR-full-mAP')]
        lines.append(f'| 均值 | {method.upper()} | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    lines += ['', '## 测试集分时长结果（两次微调均值）', '',
              '| 时长组 | 方法 | R1@0.5 | R1@0.7 | 宽度误差中位数的均值（秒） | 过宽比例 |',
              '|---|---|---:|---:|---:|---:|']
    labels = ('<5 s', '5-<10 s', '10-<20 s', '>=20 s')
    for label in labels:
        for method in METHODS:
            vals = [statistics.mean(float(tables['test', s, method][label][k]) for s in SEEDS)
                    for k in KEYS[:4]]
            lines.append(f'| {label} | {method.upper()} | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    lines += ['', '## 结论与解释限制', '',
              '- `<5秒`组在两个微调种子中，M1 的 R1@0.5 和 R1@0.7 都低于 B2。'
              '验证集也呈现同样方向；本实验不支持“增加短样本权重能改善短片段定位”的假设。',
              '- 测试集总体 R1@0.7 两个种子均有小幅提高，但短片段没有受益；长片段'
              '（>=20秒）R1@0.5、R1@0.7 两个种子均提高。整体分数不能代表短片段改善。',
              '- 短片段宽度误差和过宽比例的变化随种子而异，没有一致缓解。',
              '- 这是当前平方根权重采样方案的负结果，不能推断所有时长感知方法均无效，'
              '也不能据此确定偏差的根本原因。',
              '- 两个微调种子不足以建立稳定性或统计显著性；每个方法只有一个预训练种子。',
              '- 一条查询可能包含多个时长组的真值区间，因此分组查询数之和大于查询总数，'
              '不能按组样本数加权还原整体指标。',
              '- 宽度误差按每个模型预测与组内最大 IoU 真值匹配；不同模型可能匹配到不同'
              '真值。宽度统计是描述性诊断，不能单独证明边界误差成因。',
              '- 已检查测试结果，后续若据此修改方法，同一测试集不再是完全未使用的最终'
              '评估集；下一轮设计应依据训练集和验证集，论文需披露实验顺序。', '',
              '## 复现', '', '`python analysis/compare_b2_m1.py`', '',
              '输入为 analysis/results/{b2,m1}_seed{2023,2024}_{val,test}_by_duration/'
              ' 和对应 *_evaluation/test_metrics.json。', '',
              '图中柱高是两次微调均值，竖线覆盖两个种子的最小值与最大值；它不是置信区间。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':9,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), constrained_layout=True)
    panels = [(KEYS[0], 'R1@0.5 (%)'), (KEYS[1], 'R1@0.7 (%)'),
              (KEYS[2], 'Median signed width error (s)'), (KEYS[3], 'Overwide predictions (%)')]
    x = np.arange(4)
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for i, method in enumerate(METHODS):
            v = np.array([[float(tables['test', s, method][l][key]) for l in labels] for s in SEEDS])
            avg = v.mean(axis=0)
            ax.bar(x+(i-.5)*.35, avg, width=.35, label=method.upper(),
                   color=('#0072B2', '#D55E00')[i],
                   yerr=np.array([avg-v.min(axis=0), v.max(axis=0)-avg]), capsize=3)
        ax.set_xticks(x, ['<5', '5–<10', '10–<20', '≥20'])
        ax.set_xlabel('Ground-truth duration (s)')
        ax.set_ylabel(ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
        if key == KEYS[2]:
            ax.axhline(0, color='gray', linestyle='--', linewidth=.8)
    axes[0,0].legend(frameon=False)
    fig.suptitle('CASTELLA test: mean and range of two fine-tuning seeds')
    for ext in ('pdf', 'png'):
        fig.savefig(ROOT / 'figures' / f'b2_m1_test_duration_comparison.{ext}', dpi=300)
    plt.close(fig)
    print('\n'.join(lines[:40]))
    print('Saved:', OUT / 'report.md')


if __name__ == '__main__':
    main()
