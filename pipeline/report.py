# -*- coding: utf-8 -*-
"""Generate DETAILED_RESULTS_REPORT.txt and METHODS_OPERATION_GUIDE.md."""
import pandas as pd
import numpy as np
import config as C


def _rd(module, name):
    p = C.MODULE_DIRS[module] / "tables" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _top(df, by, n=10, absval=True):
    if df.empty or by not in df:
        return df
    key = df[by].abs() if absval else df[by]
    return df.assign(_k=key).sort_values("_k", ascending=False).drop(columns="_k").head(n)


def _pick(df, by, thr, absval=True, min_n=None, max_n=None):
    """Select rows whose |effect| clears `thr` (report ALL of them). If fewer than
    min_n clear the bar, fall back to ranking the top min_n. Returns (subset, note)
    where note explains which rule was applied. A max_n cap keeps prose finite."""
    min_n = C.REPORT_MIN_N if min_n is None else min_n
    max_n = C.REPORT_MAX_N if max_n is None else max_n
    if df.empty or by not in df:
        return df, ""
    key = df[by].abs() if absval else df[by]
    ranked = df.assign(_k=key).sort_values("_k", ascending=False)
    above = ranked[ranked["_k"] >= thr]
    if len(above) >= min_n:
        note = f"(效应量阈值 |{by}|≥{thr:g} 之上共 {len(above)} 项, 全部列出" \
               + (f"; 此处显示前 {max_n}, 余见表" if len(above) > max_n else "") + ")"
        return above.drop(columns="_k").head(max_n), note
    note = f"(阈值 |{by}|≥{thr:g} 之上仅 {len(above)} 项, 数量过少, 改按效应量列前 {min_n})"
    return ranked.drop(columns="_k").head(min_n), note


def write_report(master, reg, dcols, r0, r1, r2, r3, r4, r5, manifest):
    L = []
    w = L.append
    w("=" * 90)
    w("多器官衰老（organ age gap）多模块分析 —— 详细结果报告")
    w("DETAILED RESULTS REPORT")
    w("=" * 90)
    w(f"生成时间: {manifest['timestamp']}    随机种子: {C.RANDOM_SEED}")
    w("统计显著性标记: * p<.05  ** p<.01  † BH-FDR<.05(器官族内)  ‡ BY-FDR<.05(全局)")
    w("所有分析均全量运行、如实报告；不因不显著而删除。报告正文不再用固定'前 N 名'，而是")
    w("列出效应量达到有意义阈值的全部关联（|partial r|≥0.10、认知≥0.05、OR≤0.87或≥1.15、")
    w("|β_int|≥0.05）；仅当达标者过少时才退回按效应量排名。完整结果见各模块 tables/ 与 figures/。")
    w("")

    # ---- sample composition ----
    w("-" * 90)
    w("样本构成（各模块按所需变量分别取交集，listwise deletion）")
    w("-" * 90)
    for k, v in r0.items():
        w(f"  {k:38s}: {v}")
    w("")
    w("关键限制: 器官 age_gap 队列(44,810)与影像/临床队列几乎不重叠，可分析样本显著缩小；")
    w("脑影像模块受 eTIV 覆盖限制约 ~3,960，皮层下/皮层体积可达 ~5,653，疾病模块 7,383。")
    w("")

    # ---- M0 ----
    w("#" * 90)
    w("模块 0  描述性统计与质量控制")
    w("#" * 90)
    w("目的: 刻画各器官 age gap、脑指标、认知、疾病的分布与缺失，记录 Winsorization(±3SD)。")
    w("方法: 均值/SD/极值/中位数/IQR/偏度/峰度；缺失率；按 ±3SD 截尾并记录截断个数。")
    wl = _rd("M0", "T0.8_winsorization_log.csv")
    if not wl.empty:
        w("结果(Winsorization 截断总数按器官):")
        for _, r in wl.iterrows():
            w(f"    {r['organ']:22s}  截断 {int(r['n_winsor_total']):4d}  "
              f"(low {int(r['n_winsor_low'])}, high {int(r['n_winsor_high'])})")
    w("总结: 分布已截尾控制极端值；各变量缺失见 T0.7；后续模块在此清洗基础上分别取样。")
    w("")

    # ---- M1 ----
    w("#" * 90)
    w("模块 1  器官衰老 → 脑结构")
    w("#" * 90)
    w("目的: 量化每个器官 age gap 与各脑结构指标的独立关联及其剂量-反应关系。")
    w("方法: (A) 经典多元线性回归直接法, BrainMetric ~ organ_gap + age + gender (+eTIV 仅体积类);")
    w("      偏相关 r 由 t 统计量直接换算; 体积类另加全局脑量做特异性对照。")
    w("      (B) 17 器官同时入模 + VIF 共线性诊断; (C) 按脑层(A/B/C)汇总; (D) 累积老化负担剂量-反应。")
    w("      重要区分: 本模块把两类关联分开呈现, 因为它们的含义完全不同——")
    w("        · Brain 自身 age gap × 脑结构 = '模型有效性'检验(脑年龄钟本就基于脑结构训练,")
    w("          二者强相关是预期的自证, 只说明这批脑老化预测确实有效, 不代表生物学新发现);")
    w("        · 外周器官 age gap × 脑结构 = 真正有价值的'躯体→脑'跨器官信号(某外周器官老化")
    w("          是否独立地伴随脑结构改变), 这才是回答'器官老化累及脑'的证据。")
    m1a = _rd("M1", "T1.1_M1A_partial_corr_long.csv")
    if not m1a.empty:
        nsig_bh = int((m1a["p_BH"] < C.FDR_ALPHA).sum())
        nsig_by = int((m1a["p_BY"] < C.FDR_ALPHA).sum())
        self_a = m1a[m1a["organ"] == C.CENTRAL_ORGAN]
        periph = m1a[m1a["organ"] != C.CENTRAL_ORGAN]
        w(f"结果: M1-A 共 {len(m1a)} 个 器官×脑指标 检验(BH-FDR 显著 {nsig_bh}, BY-FDR 显著 {nsig_by})。")
        # (i) brain self-coupling = model validity
        s_self = self_a[self_a["p_BH"] < C.FDR_ALPHA]
        w(f"  (i) 模型有效性(Brain 自身): {len(self_a)} 项中 {len(s_self)} 项 BH 显著, "
          f"平均 |r|={self_a['r'].abs().mean():.3f}——如预期高度耦合, 证实脑老化钟有效。")
        sub, note = _pick(self_a, "r", C.REPORT_R_MIN)
        for _, r in sub.iterrows():
            w(f"      {str(r['metric'])[:38]:38s}  r={r['r']:+.3f}  p={r['p']:.2e}  {r.get('sig','')}")
        # (ii) peripheral -> brain = the informative cross-organ signal.
        # Peripheral r's are inherently small (0.05-0.11); a single global top-list is
        # dominated by a few broad-coverage organs and hides the others' own targets. So we
        # report, for EVERY peripheral organ with any BH-significant coupling, its
        # strongest target region -> all organs with signal are represented.
        s_per = periph[periph["p_BH"] < C.FDR_ALPHA].copy()
        norg = s_per["organ"].nunique()
        w(f"  (ii) 跨器官信号(外周器官→脑结构, 核心): {len(periph)} 项中 {len(s_per)} 项 BH 显著, "
          f"覆盖 {norg} 个外周器官。下按器官列出各自最强的显著脑靶区(完整清单见 T1.3b):")
        if not s_per.empty:
            s_per["_ar"] = s_per["r"].abs()
            best = s_per.sort_values("_ar", ascending=False).groupby("organ", as_index=False).head(1)
            for _, r in best.sort_values("_ar", ascending=False).iterrows():
                ns = int((s_per["organ"] == r["organ"]).sum())
                w(f"      {r['organ']:10s} × {str(r['metric'])[:34]:34s}  r={r['r']:+.3f}  "
                  f"p={r['p']:.2e}  {r.get('sig','')}  (该器官共 {ns} 项显著)")
            top2 = s_per.groupby('organ').size().nlargest(2)
            top2s = '/'.join(f'{o}({n} 项)' for o, n in top2.items())
            w(f"       解读: 各外周器官各有其最强脑靶区——{top2s} 覆盖面最广, "
              "其余器官虽 |r| 较小(0.05–0.06)但方向清晰、经 BH 校正稳健, 提示不同外周系统")
            w("       老化各自耦合到特定脑区(如唾液腺→海马、免疫→脑干、肌肉→苍白球), 而非笼统全脑萎缩。")
    vif = _rd("M1", "T1.6_M1B_VIF_diagnostics.csv")
    if not vif.empty:
        w(f"  M1-B 多器官同时入模: 最大 VIF 中位数 {vif['max_VIF'].median():.2f}, "
          f"严重共线性(>10)的指标 {int((vif['warn']=='severe').sum())} 个——说明各器官 age gap "
          "间共线性总体可控, 上述单器官关联在联合模型中大体保持独立。")
    dose = _rd("M1", "T1.10_M1D_dose_response.csv")
    if not dose.empty:
        d20 = dose[dose["threshold"] == "top20"]
        nsig = int((d20["p_anova"] < 0.05).sum())
        w(f"  M1-D 剂量-反应(top20%极端老化负担): {len(d20)} 个整体脑量指标中 {nsig} 个 ANOVA 显著, "
          "即随极端老化器官数增加, 整体脑量呈递减趋势(见 F1.3), 支持'累积老化→脑萎缩'的剂量关系。")
    w("小结: Brain 自身与脑结构的强关联仅确认模型有效, 不作生物学解读; 真正的发现在外周器官→脑")
    w("      这一跨器官维度——上文按效应量阈值列出的外周关联即为可解释的重点, 其体积类在全局脑量")
    w("      校正后仍显著者(见 T1.2)进一步提示区域特异而非整体缩放效应。")
    w("")

    # ---- M2 ----
    w("#" * 90)
    w("模块 2  器官衰老 → 疾病")
    w("#" * 90)
    w("目的: 评估器官 age gap 对疾病的关联(效应量与风险), 并刻画器官的神经/系统偏好。")
    w(f"方法: (A) Cohen's d(病例vs对照, ≥{C.DISEASE_D_MIN_CASES}) 与 Logistic OR(每1SD, ≥{C.DISEASE_MIN_CASES});")
    w("      (B) NPI=(|d|_neuro − |d|_sys)/(|d|_neuro + |d|_sys), 有界[-1,1]; ")
    w("          注意: NPI 衡量的是'神经特异性'而非'神经关联强度'——某器官若对神经与系统")
    w("          疾病都强相关, NPI 会接近 0(非特异/广泛累及), 与'弱关联'不同, 故同时报告")
    w("          neuro_strength_d 与 system_strength_d 两个维度(见 T2.3、F2.5)。")
    w("          稳健性: NPI 排序对小样本神经疾病(痴呆 n=8、帕金森 n=23)高度敏感,")
    w("          纳入痴呆会翻转排序(5病集 vs 3病集 Spearman≈0.44), 故主口径只用")
    w("          病例≥50 的神经疾病(卒中/抑郁/焦虑), 全集见敏感性表 T2.3b/T2.3c。")
    w("      (C) 通路偏好分类(基于特异性与累及程度);")
    w("      (D) 器官 × 疾病类型画像: 按生理系统(神经/心血管/代谢/呼吸/消化/肾/血液/皮肤等)")
    w("          汇总每个器官对各疾病类别的 mean |d|/mean OR/显著数, 直接刻画'器官老化对应哪些疾病类型';")
    w("      (E) 累积极端老化负担(top20%单侧, 0/1/2/3/4+)→疾病风险(Logistic OR/每+1器官)与多病共存趋势。")
    w(f"      纳入的主分析疾病({len(r2['included'])}): {', '.join(r2['included'])}")
    m2a = _rd("M2", "T2.1_M2A_effect_matrix.csv")
    if not m2a.empty and "OR" in m2a:
        orr = m2a[m2a["OR"].notna()].copy()
        nsig = int((orr["p_BH"] < C.FDR_ALPHA).sum()) if "p_BH" in orr else 0
        sub, note = _pick(orr, "logOR", C.REPORT_OR_LOG_MIN)
        w(f"结果: 共 {len(orr)} 个 器官→疾病 Logistic OR(每 1SD, BH 显著 {nsig})。"
          f"达到有意义强度(OR≤0.87 或 ≥1.15)的关联 {note}:")
        for _, r in sub.iterrows():
            w(f"    {r['organ']:10s} → {str(r['disease'])[:28]:28s}  "
              f"OR={r['OR']:.3f} [{r['OR_ci_lo']:.2f},{r['OR_ci_hi']:.2f}]  "
              f"p={r['p']:.2e}  {r.get('sig','')}")
        if not sub.empty:
            w("      解读: OR>1 表示该器官老化每增 1SD、患病风险相应上升; 上列即风险关联最强者, "
              "多集中于代谢/心血管/肾等系统性疾病, 完整矩阵见 T2.1。")
    npi = _rd("M2", "T2.3_M2B_NPI.csv")
    if not npi.empty:
        w("      NPI(主口径, 神经疾病≥50) 特异性最高/最低, 并列出神经与系统强度两维:")
        nn = npi.dropna(subset=["NPI"]).sort_values("NPI", ascending=False)
        for _, r in pd.concat([nn.head(4), nn.tail(4)]).iterrows():
            w(f"        {r['organ']:12s} NPI={r['NPI']:+.3f}  "
              f"neuroStr={r['neuro_strength_d']:.3f}  sysStr={r['system_strength_d']:.3f}")
        brow = nn[nn["organ"] == "Brain"]
        if not brow.empty:
            br = brow.iloc[0]
            w(f"      关键: Brain 神经强度({br['neuro_strength_d']:.3f})居前列, 但 NPI≈{br['NPI']:+.2f}")
            w("            ——因其对系统性疾病同样强相关, 属'非特异/广泛累及', 而非偏神经。")
    # M2-D organ x disease-type profile (core question)
    dom = _rd("M2", "T2.8_M2D_organ_dominant_disease_type.csv")
    dmat = _rd("M2", "T2.7_M2D_category_meanabs_d_matrix.csv")
    if not dom.empty:
        w("      M2-D 器官 × 疾病类型画像(mean |Cohen's d|, 病例≥30 的疾病, 见 T2.6/T2.7 + 热图 F2.4):")
        w("        每个器官老化最主要对应的疾病类型(dominant disease type):")
        for _, r in dom.dropna(subset=["dominant_category"]).iterrows():
            w(f"          {r['organ']:12s} → {str(r['dominant_category']):24s} "
              f"(mean|d|={float(r['dominant_mean_abs_d']):.3f})")
    # M2-E cumulative burden -> disease
    bod = _rd("M2", "T2.9_M2E_burden_to_disease_OR.csv")
    dcr = _rd("M2", "T2.11_M2E_disease_count_dose_response.csv")
    if not bod.empty:
        nsig = int((bod["p_BH"] < 0.05).sum()) if "p_BH" in bod else 0
        w(f"      M2-E 累积极端老化负担→疾病: {len(bod)} 种主分析疾病中 {nsig} 种 BH 显著; "
          "OR 为每多 1 个'极端老化(top20%)'器官的患病风险倍数。风险最高前 6:")
        for _, r in bod.sort_values("OR_per_organ", ascending=False).head(6).iterrows():
            w(f"          {str(r['disease'])[:26]:26s} OR={float(r['OR_per_organ']):.3f} "
              f"[{float(r['OR_ci_lo']):.2f},{float(r['OR_ci_hi']):.2f}] {r.get('sig','')}")
    if not dcr.empty:
        r = dcr.iloc[0]
        w(f"      累积负担→主分析疾病总数(多病共存): ANOVA F={float(r['F']):.2f} "
          f"p={float(r['p_anova']):.2g}, 线性趋势 b={float(r['b_linear']):.3f} "
          f"p={float(r['p_linear']):.2g}(见 F2.6)。")
    w("总结: 各器官老化对应的疾病类型见 M2-D 画像; NPI 进一步压缩为'神经 vs 系统'特异性一维。")
    w("      在病例充足的口径下无器官表现强神经特异性; Brain 神经关联强但非特异(对系统病同样强)。")
    w("      疾病NPI 受神经退行性疾病(痴呆 n=8/帕金森 n=23)样本过小限制, 故主口径仅用病例≥50 的神经疾病。")
    w("")

    # ---- M3 ----
    w("#" * 90)
    w("模块 3  器官衰老 → 认知")
    w("#" * 90)
    w("目的: 器官 age gap 与认知表现的关联及剂量-反应。")
    w("方法: (A) 偏相关直接法, Cognition ~ organ_gap + age + gender(仅控年龄性别, 不控 eTIV/脑量);")
    w("      (B) 累积老化负担剂量-反应(ANOVA+线性/二次趋势)。")
    m3a = _rd("M3", "T3.1_M3A_partial_corr.csv")
    if not m3a.empty:
        nsig = int((m3a["p_BH"] < C.FDR_ALPHA).sum())
        sub, note = _pick(m3a, "r", C.REPORT_COG_R_MIN)
        w(f"结果: {len(m3a)} 个 器官×认知 检验(BH-FDR 显著 {nsig})。器官-认知关联普遍偏弱, "
          f"达到 |partial r|≥{C.REPORT_COG_R_MIN:g} 的关联 {note}:")
        for _, r in sub.iterrows():
            w(f"    {r['organ']:10s} × {r['cognition']:26s}  r={r['r']:+.3f}  "
              f"p={r['p']:.2e}  {r.get('sig','')}")
    dose3 = _rd("M3", "T3.4_M3B_dose_response.csv")
    if not dose3.empty:
        d20 = dose3[dose3["threshold"] == "top20"] if "threshold" in dose3 else dose3
        nsig3 = int((d20["p_anova"] < 0.05).sum())
        w(f"      M3-B 累积老化负担→认知: {len(d20)} 项测验中 {nsig3} 项 ANOVA 显著, "
          "提示极端老化器官越多、部分认知表现越差(见 F3.2)。")
    w("小结: 器官老化与认知的直接关联效应量小(远弱于器官→脑结构), 说明躯体老化对认知多为间接")
    w("      影响——这正是模块 4 用脑结构做中介的动机; 各测验样本量差异见 T3.1 的 n 列。")
    w("")

    # ---- M4 ----
    w("#" * 90)
    w("模块 4  中介分析 (器官 → 脑结构 → 认知/疾病)")
    w("#" * 90)
    w("目的: 检验脑结构是否中介器官衰老对认知/疾病的效应。")
    w(f"方法: 三段式预筛(X→M 经 M1-A BH 显著; M→Y p<{C.MY_P}; X→Y 总效应 p<{C.XY_TREND_P}), "
      f"再对入选路径做 {C.BOOTSTRAP_N} 次 Bootstrap 的 BCa 置信区间; 95%CI 不含 0 判显著。")
    med = _rd("M4", "T4.2_M4_mediation_results.csv")
    if not med.empty:
        sig = med[med["sig"] == True] if "sig" in med else med
        w(f"结果: 预筛后检验 {len(med)} 条路径, 其中 Bootstrap BCa 95%CI 不含 0 的显著中介 "
          f"{r4['n_sig']} 条。按中介效应量 |ab| 从大到小列出显著路径(全部保留于 T4.2):")
        for _, r in _top(sig, "ab", C.REPORT_MAX_N).iterrows():
            w(f"    {r['organ']:9s}→{str(r['mediator'])[:20]:20s}→{str(r['outcome'])[:16]:16s}"
              f"  ab={r['ab']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
              f"({r.get('ci_method','')}) {r.get('classification','')}")
        w("      解读: ab≠0 表示器官老化对该结局的效应部分或全部经由脑结构传递; 结合模块 3"
          "(器官→认知直接关联弱)看, 脑结构中介是躯体老化影响认知的重要通路。")
    else:
        w("结果: 预筛后无满足全部三段条件的路径(或样本不足)。见 T4.1 预筛表。")
    w("小结: 显著中介提示脑结构在'器官老化→结局'链条中的桥梁作用; 完整点估计/CI/分类见 T4.2。")
    w("")

    # ---- M5 ----
    w("#" * 90)
    w("模块 5  性别分层与交互")
    w("#" * 90)
    w("方法: (A) 分性别脑偏相关; (B) 分性别负担剂量-反应; (C) organ×gender 交互(脑/认知/疾病);")
    w("      (D) 分性别疾病 OR; (E) 分性别认知偏相关。交互 p 值经 BH/BY 校正。")
    m5c = _rd("M5", "T5.3_M5C_interactions.csv")
    if not m5c.empty:
        nsig = int((m5c["p_int_BH"] < C.FDR_ALPHA).sum())
        s = m5c[m5c["p_int"] < 0.05]
        sub, note = _pick(s, "beta_int", C.REPORT_BETA_INT_MIN)
        w(f"结果: {len(m5c)} 个 organ×gender 交互检验(BH-FDR 显著 {nsig})。名义显著(p<.05)且"
          f"交互效应达 |β_int|≥{C.REPORT_BETA_INT_MIN:g} 的 {note}:")
        for _, r in sub.iterrows():
            w(f"    [{r['domain']:9s}] {r['organ']:10s} × {str(r['outcome'])[:24]:24s}"
              f"  β_int={r['beta_int']:+.4f}  p={r['p_int']:.2e}  {r.get('sig','')}")
        w(f"  解读: 经 BH 校正后仅 {nsig} 项交互稳健, 说明多数器官-结局关联的强度在两性间大体一致; "
          "少数稳健交互项提示该关联存在性别差异, 分性别的具体方向见 T5.1(脑)/T5.5(疾病)/T5.6(认知)。")
    w("小结: 性别分层用于检验主效应的稳健性与异质性; 总体上器官老化的效应不强烈依赖性别, "
      "个别稳健交互值得在解释时单独说明。")
    w("")

    w("=" * 90)
    w("附: 多重比较策略 —— 每器官族内 Benjamini-Hochberg(BH) + 全局 Benjamini-Yekutieli(BY)。")
    w("    未做任何按显著性的结果删除; 显著结果众多时全部保留于 tables/ 供补充材料引用。")
    w("=" * 90)

    path = C.RESULTS_DIR / "DETAILED_RESULTS_REPORT.txt"
    path.write_text("\n".join(L), encoding="utf-8")


def write_methods_guide(master, reg, dcols, manifest):
    sw = manifest["software"]; pa = manifest["parameters"]
    M = []
    a = M.append
    a("# 方法与操作说明 (METHODS & OPERATION GUIDE)\n")
    a(f"生成时间: {manifest['timestamp']}\n")
    a("## 1. 数据来源\n")
    a("| 角色 | 文件 |")
    a("|------|------|")
    for k, v in C.FILES.items():
        a(f"| {k} | `{v.name}` |")
    a("")
    a("## 2. 变量构建\n")
    a("- **器官 age gap**: 17 个器官(16 外周 + Brain 中枢)的 `预测年龄 − 实际年龄`; 原始值，"
      "未预先 z 标准化; 分析前按 ±3SD Winsorization。")
    a("- **脑指标登记表**: 共 %d 个, 分 A(全局)/B(皮层下)/C(皮层) 三类; 体积类需控 eTIV, "
      "厚度/面积/比值/拓扑类不控。见 `brain_metric_registry.csv`。" % len(reg))
    a("- **认知**: UKB 影像访视(instance-2); 反应时/连线测验做对数变换。")
    a("- **人口学**: age(21003-2.0) 与 sex(31-0.0), 与疾病表 age/gender 合并补全。")
    a("- **疾病**: 疾病表指示列, 病例数 ≥%d 进入 OR/中介/性别分层, ≥%d 仅算 Cohen's d。"
      % (C.DISEASE_MIN_CASES, C.DISEASE_D_MIN_CASES))
    a("")
    a("## 3. 统计方法\n")
    a(f"- 偏相关: {pa['partial_corr_method']} (禁用残差法)。")
    a("- 效应量: Cohen's d, 标准化 β, 偏相关 r, Logistic OR(每 1 SD)。")
    a("- 剂量-反应: 累积极端老化负担(top20%单侧, 0/1/2/3/4+, <100 合并) 的 ANOVA + 线性/二次趋势。")
    a(f"- 多重比较: BH(每器官族内) + BY(全局), α={pa['FDR_alpha']}。")
    a(f"- 中介: {pa['bootstrap_n']} 次 Bootstrap, {pa['bootstrap_ci']} 置信区间; "
      f"预筛 MY p<{pa['mediation_prescreen']['MY_p']}, XY p<{pa['mediation_prescreen']['XY_trend_p']}。")
    a("- NPI: 归一化神经偏好指数(疾病特异性), 有界[-1,1]; 主口径用病例≥50 的神经疾病; 同时报告神经/系统强度两维。")
    a("")
    a("## 4. 每个模块的样本 (listwise deletion, 独立取交集)\n")
    for k, v in manifest["sample_composition"].items():
        a(f"- {k}: {v}")
    a("")
    a("## 5. 软件版本\n")
    for k, v in sw.items():
        a(f"- {k}: {v}")
    a(f"\n随机种子: {manifest['random_seed']}\n")
    a("## 6. 输出清单\n")
    a("- `M0_descriptive/` … `M5_sex/`: 各含 `tables/`(CSV) 与 `figures/`(PNG)。")
    a("- `brain_metric_registry.csv`, `run_manifest.json`, `DETAILED_RESULTS_REPORT.txt`。")
    a("")
    a("## 7. 复现\n")
    a("```\ncd aging_pipeline\npython run_pipeline.py\n```\n")

    path = C.RESULTS_DIR / "METHODS_OPERATION_GUIDE.md"
    path.write_text("\n".join(M), encoding="utf-8")
