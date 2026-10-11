"""CPU-only validation diagnostics; prefer full predictions, fall back to Top-1 CSV.

No test predictions are used. Oracle transformations use labels and are diagnostics,
not deployable model results. Quantized Top-1 CSVs cannot recover pre-rounding output.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results/phase2'
FIG = ROOT / 'analysis/figures/phase2'
BINS = {'All': lambda w: True, '<5 s': lambda w: w < 5,
        '5-<10 s': lambda w: 5 <= w < 10, '10-<20 s': lambda w: 10 <= w < 20,
        '>=20 s': lambda w: w >= 20}


def read_jsonl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def iou(p, g):
    overlap = max(0., min(p[1], g[1]) - max(p[0], g[0]))
    union = p[1]-p[0] + g[1]-g[0] - overlap
    return overlap / union if union > 0 else 0.


def center(p):
    return (p[0]+p[1])/2


def width(p):
    return p[1]-p[0]


def choose(p, windows, nearest=False):
    # All-zero-IoU ties use nearest center, then start/end, never input order.
    return min(windows, key=lambda g: ((0 if nearest else -iou(p,g)),
                                      abs(center(p)-center(g)), g[0], g[1]))


def transform(c, w, rounded):
    # Mirrors official clip_ts [0,300] followed by torch.round (ties to even).
    p = [min(300., max(0., c-w/2)), min(300., max(0., c+w/2))]
    return [float(round(v)) for v in p] if rounded else p


def write_csv(name, rows):
    if not rows:
        return
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def recall(values, threshold):
    return 100*sum(v >= threshold for v in values)/len(values)


def load_predictions(method, seed, gt):
    raw = ROOT / f'analysis/inputs/phase2/{method}_seed{seed}/val_predictions.jsonl'
    if raw.exists() and raw.stat().st_size > 0:
        items = read_jsonl(raw)
        assert len({x['qid'] for x in items}) == len(items)
        pred = {x['qid']:x['pred_relevant_windows'] for x in items}
        full = True
    else:
        raw = ROOT / f'analysis/results/{method}_seed{seed}_val_by_duration/per_query.csv'
        pred = {}
        with raw.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                p = [float(r[k]) for k in ('pred_start','pred_end','top1_score')]
                if r['qid'] in pred:
                    assert pred[r['qid']] == [p], 'Repeated groups disagree on Top-1'
                pred[r['qid']] = [p]
        full = False
    assert pred.keys() == gt.keys(), 'Missing/extra qids'
    for qid, ps in pred.items():
        assert ps and all(np.isfinite(p).all() and 0 <= p[0] <= p[1] <= 300 for p in ps)
        # Scores are rounded to four decimals; preserve existing order within ties.
        assert all(ps[i][2] >= ps[i+1][2] for i in range(len(ps)-1))
    return pred, {'method':method, 'seed':seed, 'input':str(raw.relative_to(ROOT)),
                  'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
                  'queries':len(pred), 'full_candidates':full,
                  'min_candidates':min(map(len,pred.values())),
                  'max_candidates':max(map(len,pred.values()))}


def analyze_model(method, seed, gt):
    predictions, provenance = load_predictions(method, seed, gt)
    details, summaries, oracle_rows, rank_rows, scaling_rows = [], [], [], [], []
    for label, belongs in BINS.items():
        records = []
        oracle = {(mode, rounded, pairing): [] for mode in ('center','width','both')
                  for rounded in (False,True) for pairing in ('best_over_gt','fixed_gt')}
        covers = {(k,t):[] for k in (1,3,5,10) for t in (.5,.7)}
        scales = {a:[] for a in (.5,.75,1.,1.25)}
        for qid, item in gt.items():
            windows = [g for g in item['relevant_windows'] if belongs(width(g))]
            if not windows:
                continue
            ps = predictions[qid]
            p = ps[0][:2]
            g = choose(p,windows)
            near = choose(p,windows,nearest=True)
            best = max(iou(p,w) for w in windows)
            has_overlap = best > 0
            center_inside = any(w[0] <= center(p) <= w[1] for w in windows)
            category = ('correct' if best >= .7 else 'center_inside' if center_inside
                        else 'overlap_center_outside' if has_overlap else 'no_overlap')
            r = dict(method=method,seed=seed,duration_bin=label,qid=qid,vid=item['vid'],
                     query=item['query'],gt_count=len(windows),top1_iou=best,
                     pred_start=p[0],pred_end=p[1],pred_width=width(p),
                     matched_start=g[0],matched_end=g[1],matched_width=width(g),
                     category=category,overlap=has_overlap,center_inside=center_inside,
                     group_only=all(belongs(width(w)) for w in item['relevant_windows']),
                     any_gt_iou=max(iou(p,w) for w in item['relevant_windows']),
                     width_error=width(p)-width(g),width_ratio=width(p)/width(g),
                     center_error=abs(center(p)-center(g)),
                     relative_center_error=abs(center(p)-center(g))/width(g),
                     nearest_center_error=abs(center(p)-center(near)))
            records.append(r)
            details.append(r)
            for (mode, rounded, pairing), values in oracle.items():
                eligible = windows if pairing == 'best_over_gt' else [g]
                changed = [transform(center(w) if mode in ('center','both') else center(p),
                                     width(w) if mode in ('width','both') else width(p), rounded)
                           for w in eligible]
                values.append(max(iou(cp,w) for cp in changed for w in windows))
            for alpha, values in scales.items():
                cp = transform(center(p),width(p)*alpha,True)
                values.append(max(iou(cp,w) for w in windows))
            if provenance['full_candidates']:
                for (k,t), values in covers.items():
                    values.append(max(iou(cp,w) for cp in ps[:k] for w in windows) >= t)
        baseline = [r['top1_iou'] for r in records]
        overlap_records = [r for r in records if r['overlap']]
        sr = dict(method=method,seed=seed,duration_bin=label,n=len(records),
                  r1_05=recall(baseline,.5),r1_07=recall(baseline,.7))
        for cat in ('correct','center_inside','overlap_center_outside','no_overlap'):
            sr[cat+'_percent'] = 100*sum(r['category']==cat for r in records)/len(records)
        sr['center_inside_any_percent'] = 100*sum(r['center_inside'] for r in records)/len(records)
        sr['center_error_median'] = statistics.median(r['center_error'] for r in records)
        sr['nearest_center_error_median'] = statistics.median(r['nearest_center_error'] for r in records)
        sr['width_error_median'] = statistics.median(r['width_error'] for r in records)
        sr['overwide_percent'] = 100*sum(r['width_error']>0 for r in records)/len(records)
        sr['overlap_n'] = len(overlap_records)
        sr['overlap_width_error_median'] = statistics.median(r['width_error'] for r in overlap_records) if overlap_records else float('nan')
        sr['overlap_overwide_percent'] = 100*sum(r['width_error']>0 for r in overlap_records)/len(overlap_records) if overlap_records else float('nan')
        group_only = [r for r in records if r['group_only']]
        sr['group_only_n'] = len(group_only)
        sr['group_only_no_overlap_percent'] = 100*sum(not r['overlap'] for r in group_only)/len(group_only) if group_only else float('nan')
        sr['no_overlap_any_gt_percent'] = 100*sum(r['any_gt_iou']==0 for r in records)/len(records)
        summaries.append(sr)
        # Verify that revised tie-breaking does not change recall.
        old = ROOT/f'analysis/results/{method}_seed{seed}_val_by_duration/summary.csv'
        with old.open(encoding='utf-8-sig',newline='') as f:
            oldrow = next(r for r in csv.DictReader(f) if r['duration_bin']==label)
        assert abs(sr['r1_07']-float(oldrow['r1_at_0.7_percent'])) < 1e-8
        for (mode,rounded,pairing), values in oracle.items():
            oracle_rows.append(dict(method=method,seed=seed,duration_bin=label,mode=mode,
                                    rounded=rounded,pairing=pairing,n=len(values),
                                    r1_05=recall(values,.5),r1_07=recall(values,.7),
                                    delta_r1_07=recall(values,.7)-sr['r1_07'],
                                    improved=sum(v>b for v,b in zip(values,baseline)),
                                    worsened=sum(v<b for v,b in zip(values,baseline))))
        for alpha, values in scales.items():
            scaling_rows.append(dict(method=method,seed=seed,duration_bin=label,alpha=alpha,
                                     r1_05=recall(values,.5),r1_07=recall(values,.7)))
        if provenance['full_candidates']:
            for (k,t), values in covers.items():
                rank_rows.append(dict(method=method,seed=seed,duration_bin=label,k=k,
                                      threshold=t,coverage_percent=100*sum(values)/len(values)))
    return details, summaries, oracle_rows, rank_rows, scaling_rows, provenance


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    gt_items = read_jsonl(ROOT/'data/castella_val_release.jsonl')
    gt = {r['qid']:r for r in gt_items}
    assert len(gt)==len(gt_items)
    combined = [[],[],[],[],[],[]]
    for method in ('b2','m1'):
        for seed in (2023,2024):
            result = analyze_model(method,seed,gt)
            for i in range(5):
                combined[i].extend(result[i])
            combined[5].append(result[5])
    names = ('per_query.csv','geometry_summary.csv','oracle_summary.csv',
             'candidate_coverage.csv','width_scaling_exploratory.csv')
    for name, rows in zip(names,combined):
        write_csv(name,rows)
    (OUT/'provenance.json').write_text(json.dumps(combined[5],ensure_ascii=False,indent=2),encoding='utf-8')

    # Ground-truth rounding check: this is not raw model output or a feature ceiling.
    quant = []
    for label,belongs in BINS.items():
        ws = [w for r in gt.values() for w in r['relevant_windows'] if belongs(width(w))]
        scores = [iou(transform(center(w),width(w),True),w) for w in ws]
        quant.append(dict(duration_bin=label,gt_windows=len(ws),min_self_iou=min(scores),
                          self_r1_07=recall(scores,.7)))
    write_csv('annotation_rounding_check.csv',quant)

    s,o = combined[1],combined[2]
    lines = ['# 第二阶段 CPU 诊断结果', '',
             '仅使用CASTELLA验证集，四组模型都是第一阶段最佳验证checkpoint。', '',
             '## 短于5秒的Top-1错误类型', '',
             '| 模型 | seed | R1@0.7 | 中心在目标内但失败 | 有重叠但中心在外 | 无重叠 | 最近中心误差中位数（秒） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in s:
        if r['duration_bin']=='<5 s':
            lines.append(f"| {r['method'].upper()} | {r['seed']} | {r['r1_07']:.2f}% | {r['center_inside_percent']:.2f}% | {r['overlap_center_outside_percent']:.2f}% | {r['no_overlap_percent']:.2f}% | {r['nearest_center_error_median']:.2f} |")
    lines += ['', '## 短片段真实值替换（1秒舍入，理想真值选择）', '',
              '| 模型 | seed | 原始R1@0.7 | 仅替换中心 | 仅替换宽度 | 两者替换 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in s:
        if r['duration_bin']!='<5 s':
            continue
        def ov(mode):
            return next(x['r1_07'] for x in o if x['method']==r['method'] and x['seed']==r['seed']
                        and x['duration_bin']=='<5 s' and x['mode']==mode and x['rounded'] and x['pairing']=='best_over_gt')
        lines.append(f"| {r['method'].upper()} | {r['seed']} | {r['r1_07']:.2f}% | {ov('center'):.2f}% | {ov('width'):.2f}% | {ov('both'):.2f}% |")
    lines += ['', '## 数据来源与限制', '',
              '- 最大IoU匹配在零重叠时使用最近中心打破平局；R1与已有统计逐组核对一致。',
              '- 真实值替换使用答案和理想真值选择，只是诊断，不能当作可运行方法提升。'
              '同时输出固定匹配、连续时间与1秒舍入结果，记录单因素替换的改善/恶化数量。',
              '- 固定宽度缩放只在验证集探索，对所有查询使用同一alpha，未读取真值时长决定是否缩放。',
              '- 真实标注舍入检查只能说明标签的量化情况，不能恢复模型舍入前预测或推断特征时间分辨率。']
    lines += ['', '## 多真值敏感性检查', '',
              '| 模型 | seed | 仅含短目标的查询数 | 这些查询无重叠比例 | 短片段组中与任意真值均无重叠比例 |',
              '|---|---:|---:|---:|---:|']
    for r in s:
        if r['duration_bin']=='<5 s':
            lines.append(f"| {r['method'].upper()} | {r['seed']} | {r['group_only_n']} | {r['group_only_no_overlap_percent']:.2f}% | {r['no_overlap_any_gt_percent']:.2f}% |")
    lines += ['', '## 初步解释与下一步', '',
              '- 真实中心替换的提升大于真实宽度替换；大量Top-1与短目标不重叠，支持优先检查位置和候选选择。'
              '这是条件诊断，不是位置误差占比的因果证明，两种修正空间不能相加。',
              '- 固定宽度缩放0.5、0.75、1.0、1.25没有显示跨两个B2种子一致的短片段R1@0.7提升；'
              '统一缩短预测不应作为首选路线。',
              '- 当前数据可检查量化后的真值：973个验证真值窗口经1秒舍入均保持原样。'
              '这不能说明模型舍入无害或离线特征有足够的时间分辨率。',
              '- 一条查询可能同时包含短、长目标；组内无重叠不等于没找到任何正确声音。'
              '应结合仅含短目标的敏感性检查解释。',
              '- 下一项应补齐完整候选列表的Top-k覆盖分析：若正确候选存在但排得低，研究排序；'
              '若所有候选都失败，研究候选生成或位置监督。此时不启动新的训练。', '',
              '复现：`python analysis/phase2_cpu_diagnostics.py`。']
    missing = [f"{x['method']}-{x['seed']}" for x in combined[5] if not x['full_candidates']]
    lines.append('- 缺少完整候选列表：' + ', '.join(missing) + '。这些模型只完成Top-1诊断，Top-k覆盖尚未执行。' if missing else '- 四组完整候选覆盖分析已执行。')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    shorts=[r for r in s if r['duration_bin']=='<5 s']
    x=np.arange(len(shorts))
    labels=[f"{r['method'].upper()}\n{r['seed']}" for r in shorts]
    bottom=np.zeros(len(shorts))
    for cat,color,title in [('correct','#009E73','IoU >= 0.7'),('center_inside','#0072B2','Center inside, failed'),
                             ('overlap_center_outside','#E69F00','Overlap, center outside'),('no_overlap','#D55E00','No overlap')]:
        v=np.array([r[cat+'_percent'] for r in shorts])
        axes[0].bar(x,v,bottom=bottom,label=title,color=color)
        bottom+=v
    axes[0].set_ylabel('Queries (%)')
    axes[0].set_title('Short targets: Top-1 error geometry')
    axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2)
    for j,(mode,title) in enumerate([('original','Original'),('center','True center'),('width','True width')]):
        vals=[]
        for r in shorts:
            val=r['r1_07'] if mode=='original' else next(z['r1_07'] for z in o if z['method']==r['method'] and z['seed']==r['seed'] and z['duration_bin']=='<5 s' and z['mode']==mode and z['rounded'] and z['pairing']=='best_over_gt')
            vals.append(val)
        axes[1].bar(x+(j-1)*.24,vals,width=.24,label=title)
    axes[1].set_ylabel('R1@0.7 (%)')
    axes[1].set_title('Label-dependent oracle diagnostics')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,labels)
        ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','png'):
        fig.savefig(FIG/f'short_geometry_oracles.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    # Deterministic case selection: first three qids per category, no cherry-picking.
    cases=[]
    base=[r for r in combined[0] if r['method']=='b2' and r['seed']==2023 and r['duration_bin']=='<5 s']
    for category in ('correct','center_inside','overlap_center_outside','no_overlap'):
        cases.extend(sorted((r for r in base if r['category']==category),key=lambda r:r['qid'])[:3])
    write_csv('selected_cases.csv',cases)
    with PdfPages(FIG/'selected_cases.pdf') as pdf:
        for offset in range(0,len(cases),3):
            page=cases[offset:offset+3]
            fig, axes=plt.subplots(len(page),1,figsize=(10,7.2),constrained_layout=True,squeeze=False)
            for ax,r in zip(axes.flat,page):
                item=gt[r['qid']]
                for g in item['relevant_windows']:
                    ax.plot(g,[1,1],linewidth=6,color='#009E73' if width(g)<5 else '#999999',solid_capstyle='butt')
                ax.plot([r['pred_start'],r['pred_end']],[0,0],linewidth=6,color='#D55E00',solid_capstyle='butt')
                ax.set_xlim(0,item['duration'])
                ax.set_ylim(-.5,1.5)
                ax.set_yticks([0,1],['Top-1','GT'])
                ax.set_title(textwrap.fill(f"{r['qid']} | {r['category']} | IoU={r['top1_iou']:.2f} | {r['query']}",110),fontsize=8)
                ax.set_xlabel('Time (s)')
            fig.suptitle('B2 seed 2023 validation cases: green = short GT, gray = other GT')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)
    print('Completed four models; summaries:')
    for r in shorts:
        print(r)
    print('Report:',OUT/'report.md')


if __name__=='__main__':
    main()
