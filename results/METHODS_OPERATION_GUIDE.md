# 方法与操作说明 (METHODS & OPERATION GUIDE)

生成时间: 2026-07-08 14:41:15

## 1. 数据来源

| 角色 | 文件 |
|------|------|
| age_gap | `merged_age_gap.csv` |
| brain_global | `全局脑指标.csv` |
| brain_subcort | `脑结构指标.csv` |
| brain_cortex | `脑结构指标2.csv` |
| clinical | `Clinical_vars.csv` |
| cognition | `Cognition9.csv` |
| disease | `final_analysis_table.csv` |

## 2. 变量构建

- **器官 age gap**: 18 个器官(17 外周 + Brain 中枢)的 `预测年龄 − 实际年龄`; 原始值，未预先 z 标准化; 分析前按 ±3SD Winsorization。
- **脑指标登记表**: 共 111 个, 分 A(全局)/B(皮层下)/C(皮层) 三类; 体积类需控 eTIV, 厚度/面积/比值/拓扑类不控。见 `brain_metric_registry.csv`。
- **认知**: UKB 影像访视(instance-2); 反应时/连线测验做对数变换。
- **人口学**: age(21003-2.0) 与 sex(31-0.0), 与疾病表 age/gender 合并补全。
- **疾病**: 疾病表指示列, 病例数 ≥50 进入 OR/中介/性别分层, ≥30 仅算 Cohen's d。

## 3. 统计方法

- 偏相关: direct multiple OLS regression (no residualisation) (禁用残差法)。
- 效应量: Cohen's d, 标准化 β, 偏相关 r, Logistic OR(每 1 SD)。
- 剂量-反应: 累积极端老化负担(top20%单侧, 0/1/2/3/4+, <100 合并) 的 ANOVA + 线性/二次趋势。
- 多重比较: BH(每器官族内) + BY(全局), α=0.05。
- 中介: 200 次 Bootstrap, BCa 置信区间; 预筛 MY p<0.05, XY p<0.1。

## 4. 每个模块的样本 (listwise deletion, 独立取交集)

- N_total_age_gap: 44810
- N_with_demographics: 7954
- N_brain_global(eTIV): 3960
- N_brain_global(BrainSegVol): 3960
- N_brain_subcortical: 5653
- N_disease: 7383
- N_cog_FluidIntelligence: 7282
- N_cog_ReactionTime: 7352
- N_cog_NumericMemory: 3796
- N_cog_TowerRearranging: 3653
- N_cog_PairedAssociateLearning: 6152
- N_cog_TrailMaking_Numeric: 6095
- N_cog_TrailMaking_Alphanumeric: 5949

## 5. 软件版本

- python: 3.12.8
- numpy: 2.4.6
- pandas: 3.0.3
- scipy: 1.18.0
- statsmodels: 0.14.6
- matplotlib: 3.11.0
- seaborn: 0.13.2

随机种子: 12345

## 6. 输出清单

- `M0_descriptive/` … `M5_sex/`: 各含 `tables/`(CSV) 与 `figures/`(PNG)。
- `brain_metric_registry.csv`, `run_manifest.json`, `DETAILED_RESULTS_REPORT.txt`。

## 7. 复现

```
cd aging_pipeline
python run_pipeline.py
```
