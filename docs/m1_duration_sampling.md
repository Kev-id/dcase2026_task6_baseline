# M1：温和时长匹配预训练

## 研究问题

Clotho-Moment与CASTELLA训练标注的片段时长分布差异很大。M1只改变Clotho-Moment预训练阶段的样本抽样概率，用来检验：让预训练数据更接近真实训练数据的时长分布，能否减轻后续CASTELLA定位中的时长相关偏差。

模型结构、特征、损失函数、batch size、学习率、训练轮数和随机种子均与原始预训练B2保持一致。

## 方法

时长区间固定为`[0,5)`、`[5,10)`、`[10,20)`和`[20,+∞)`秒。只使用CASTELLA训练集估计目标分布，不使用验证集或测试集。

对位于区间`b`的Clotho-Moment样本，抽样权重为：

```text
w_b = (p_CASTELLA_train(b) / p_Clotho_train(b)) ^ alpha
```

主实验取`alpha=0.5`，即平方根温度的重要性采样。每轮仍然抽取32,694个样本，使用有放回抽样。

| 时长区间 | Clotho原始占比 | CASTELLA目标占比 | M1权重 | M1期望占比 | 第0轮实测占比 |
|---|---:|---:|---:|---:|---:|
| `<5 s` | 1.21% | 54.72% | 6.7128 | 12.79% | 13.15% |
| `5-<10 s` | 2.66% | 21.05% | 2.8141 | 11.74% | 11.75% |
| `10-<20 s` | 43.57% | 12.93% | 0.5447 | 37.23% | 37.15% |
| `>=20 s` | 52.56% | 11.30% | 0.4638 | 38.24% | 37.96% |

## 为什么不做精确匹配

精确匹配对应`alpha=1`。它会给`<5 s`样本45.06倍权重，使仅有的397条短样本在每轮平均出现约45次。虽然397条样本来自394段不同音频，但只有147种不同文本，过拟合和文本重复风险很高。

`alpha=0.5`将短片段在每轮的平均出现次数降到约10.8次，并仍然把两个短时长区间的总占比从3.87%提高到约24.5%。精确匹配结果保留为压力测试，不作为首个主实验。

M1暂不加入边界抖动、特征噪声或文本增强。这样B2与M1之间只有采样分布这一个变量；如果M1出现明显过拟合，再把增强作为独立的M2消融实验。

## 复现命令

验证理论分布和固定随机种子的实际抽样分布：

```bash
python analysis/validate_m1_sampling.py \
  --power 0.5 \
  --output analysis/results/m1_sampling_tempered_validation.json
```

正式训练：

```bash
python src/train.py \
  --config experiments/configs/clotho_duration_matched_m1.yml
```

后台运行：

```bash
nohup python src/train.py \
  --config experiments/configs/clotho_duration_matched_m1.yml \
  > m1_pretraining.log 2>&1 < /dev/null &
```

自动关机后恢复：

```bash
nohup python src/train.py \
  --config experiments/configs/clotho_duration_matched_m1.yml \
  --resume-training experiments/m1_clotho_duration_tempered/latest_checkpoint.pth \
  > m1_pretraining_resume.log 2>&1 < /dev/null &
```

输出目录为`experiments/m1_clotho_duration_tempered/`。采样器使用`seed + epoch`生成每轮索引，因此同一轮的抽样次序在重新启动后保持一致。

## 后续比较

M1预训练完成后，应使用与B2完全相同的CASTELLA微调配置和随机种子。主要比较总体R1/mAP、四个时长区间的R1、宽度误差中位数和预测过宽比例。
