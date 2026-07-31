SPLITS = [ 'train', 'val', 'test']
SETTING2TRUE_OR_FALSE = {"True": True, "False": False}
INT2SOURCE = {0: "cell", 1: "mouse", 2: "patient"}

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import copy
import random
import math
import os
from sklearn.metrics import roc_auc_score

from plato.utils.torch_utils import dt, get_l1_reg, get_l2_reg
# from plato.baseline.evaluator import RegEval
from plato.baseline.evaluator import RegEval, MultilabelEval
from plato.utils.py_utils import flatten_nested_dict

# === 预测层硬规则拦截核心逻辑 (加强版 Method A) ===
def apply_hard_rules(x, pred, task_type, feat_indices):
    """在预测输出前，根据医学指南强行修正概率 (Logits)"""
    if task_type != "multilabel":
        return pred
        
    pred_hijacked = pred.clone()
    
    # 获取列号 (安全获取，找不到为 -1)
    def get_val(key):
        idx = feat_indices.get(key, -1)
        if idx != -1 and x.shape[1] > idx:
            return x[:, idx] == 1.0
        return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    # 提取当前 Batch 患者的各种特征状态 (True / False)
    has_ecg      = get_val('ECG_ISCHEMIA')
    has_crush    = get_val('NATURE_CRUSH')
    has_suff     = get_val('NATURE_SUFF')
    has_nitro    = get_val('NITRO_RELIEF')
    has_sweat    = get_val('SWEAT_YES')
    has_weak     = get_val('WEAK_LEGS')
    has_dur_long = get_val('DUR_LONG1') | get_val('DUR_LONG2')
    has_no_pain  = get_val('NO_PAIN')
    has_risk     = get_val('RISK_SMOKE')

    # --- 规则 1：ST段/非ST段 心肌梗死 (STEMI/NSTEMI, 索引 3, 4) ---
    # 指南：压榨痛 + 大汗 + 持续时间长
    mask_mi = has_crush | has_sweat | has_dur_long
    pred_hijacked[mask_mi, 3] += 3.0
    pred_hijacked[mask_mi, 4] += 3.0
    pred_hijacked[mask_mi, 1] += 3.0 # 父类：冠心病

    # --- 规则 2：稳定型心绞痛 (SA, 索引 5) ---
    # 指南：硝酸甘油能快速缓解
    pred_hijacked[has_nitro, 5] += 4.0
    pred_hijacked[has_nitro, 1] += 3.0 

    # --- 规则 3：隐匿性/无症状型心肌缺血 (SMI, 索引 6) ---
    # 指南：没有胸痛，但心电图有缺血
    mask_smi = has_no_pain & has_ecg
    pred_hijacked[mask_smi, 6] += 5.0
    pred_hijacked[mask_smi, 1] += 3.0

    # --- 规则 4：缺血性心肌病 (ICM, 索引 7) ---
    # 指南：心电图缺血 + 憋闷感/濒死感/双下肢无力
    mask_icm = has_ecg & (has_suff | has_weak)
    pred_hijacked[mask_icm, 7] += 3.0
    pred_hijacked[mask_icm, 1] += 3.0

    # --- 规则 5：基本危险因素累加 (CHD, 索引 1) ---
    # 指南：有吸烟等危险因素，基础发病率上升
    pred_hijacked[has_risk, 0] += 1.0 # 冠状动脉粥样硬化
    pred_hijacked[has_risk, 1] += 1.0 # 冠心病
    
    # --- 规则 6：父子层级底线传导 ---
    sub_disease_indices = [2, 3, 4, 5, 6, 7] 
    max_sub_logits, _ = torch.max(pred_hijacked[:, sub_disease_indices], dim=1)
    mask_logic_error = (max_sub_logits > 0) & (max_sub_logits > pred_hijacked[:, 1])
    pred_hijacked[mask_logic_error, 1] = max_sub_logits[mask_logic_error] + 1.0

    return pred_hijacked
# ==========================================================

def set_random_seeds(seed):
    print(f"seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device(device_in):
    device = f'cuda:{device_in}' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    device = torch.device(device)
    torch.set_num_threads(1)
    return device

# def get_task(sourceint, source2task):
#     return source2task[INT2SOURCE[sourceint]]

def get_task(sourceint, source2task):
    # 1. 优先尝试直接用整数 ID 查找 (适配自定义数据 {0: "multilabel"})
    if sourceint in source2task:
        return source2task[sourceint]
    
    # 2. 如果找不到，尝试转换成字符串名称查找 (适配原始 PLATO 数据 {"cell": "numeric"})
    if sourceint in INT2SOURCE:
        source_name = INT2SOURCE[sourceint]
        if source_name in source2task:
            return source2task[source_name]
            
    # 3. 实在找不到抛出详细错误
    raise KeyError(f"Could not find task for source ID {sourceint}. source2task keys: {list(source2task.keys())}")

# === [新增] 计算类别权重的辅助函数 ===
def compute_class_weights(loader, device):
    print("Calculating class weights from training data to handle imbalance...")
    all_y = []
    # 遍历一遍 DataLoader 收集所有标签
    for _, y, _, _, _ in loader:
        all_y.append(y)
    
    # 拼接: (N_total, Num_classes)
    all_y = torch.cat(all_y, dim=0).float()
    
    # 统计正负样本数
    num_pos = all_y.sum(dim=0)
    num_neg = all_y.size(0) - num_pos
    
    # 计算权重: Neg / Pos
    # 加上 1e-6 防止除以 0
    pos_weight = num_neg / (num_pos + 1e-6)
    
    # 如果某个类全是负样本(Pos=0)，权重设为1(不加权)，避免 Inf
    pos_weight = torch.where(num_pos == 0, torch.ones_like(pos_weight), pos_weight)
    
    print(f"  - Class Positive Counts: {num_pos.cpu().numpy().astype(int)}")
    print(f"  - Computed Pos Weights:  {pos_weight.cpu().numpy()}")
    
    return pos_weight.to(device)
# ====================================

def compute_multilabel_feedback(y_true, y_prob, threshold=0.5):
    """计算每个疾病标签的反馈指标，用于调节自适应采样强度。

    自适应采样只需要标签级错误方向：FPR 偏高表示阴性样本容易被误判为阳性；
    FNR 偏高表示需要抑制阴性增强，以保护阳性召回率。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    eps = 1e-8

    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    tn = ((y_true == 0) & (y_pred == 0)).sum(axis=0).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    auroc = np.full(y_true.shape[1], np.nan, dtype=float)
    for label_idx in range(y_true.shape[1]):
        try:
            auroc[label_idx] = roc_auc_score(y_true[:, label_idx], y_prob[:, label_idx])
        except ValueError:
            auroc[label_idx] = np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "pos_count": tp + fn,
        "neg_count": tn + fp,
    }

def classify_disease_types(feedback, args, prevalence):
    """根据验证集表现和阳性占比给每个疾病分型，供采样和阈值优化共同使用。"""
    prevalence = np.asarray(prevalence, dtype=float)
    n_labels = len(prevalence)

    recall = feedback["recall"]
    fnr = feedback["fnr"]
    specificity = feedback["specificity"]
    fpr = feedback["fpr"]
    f1 = feedback.get("f1", np.zeros(n_labels, dtype=float))
    auroc = np.nan_to_num(feedback.get("auroc", np.full(n_labels, np.nan)), nan=-1.0)

    low_prev = prevalence <= getattr(args, "adaptive_bidir_low_prevalence_threshold", 0.25)
    high_prev = prevalence >= getattr(args, "adaptive_bidir_high_prevalence_threshold", 0.65)
    miss_problem = (fnr >= args.adaptive_bidir_tau_fnr) | (recall <= args.adaptive_bidir_tau_recall)
    fp_problem = (fpr >= args.adaptive_bidir_tau_fpr) | (specificity <= args.adaptive_bidir_tau_specificity)

    has_enough_samples = (
        (feedback["pos_count"] >= getattr(args, "disease_typed_stable_min_pos", 3))
        & (feedback["neg_count"] >= getattr(args, "disease_typed_stable_min_neg", 3))
    )
    stable_protected = (
        has_enough_samples
        & (auroc >= getattr(args, "disease_typed_stable_auroc_threshold", 0.85))
        & (f1 >= getattr(args, "disease_typed_stable_f1_threshold", 0.60))
        & (recall >= getattr(args, "disease_typed_stable_recall_threshold", 0.70))
        & (specificity >= getattr(args, "disease_typed_stable_specificity_threshold", 0.70))
    )

    disease_type = np.full(n_labels, "stable", dtype=object)
    active = ~stable_protected
    disease_type[active & miss_problem & ~fp_problem] = "negative_bias"
    disease_type[active & fp_problem & ~miss_problem] = "positive_bias"
    disease_type[active & miss_problem & fp_problem] = "bidirectional_confusion"
    disease_type[active & low_prev & miss_problem] = "low_prevalence_miss"
    disease_type[active & high_prev & fp_problem] = "high_prevalence_false_positive"
    disease_type[stable_protected] = "stable_protected"

    return disease_type

def _binary_metrics_for_threshold(y_true_col, y_prob_col, threshold):
    """计算单个疾病在指定阈值下的二分类指标。"""
    y_pred_col = (y_prob_col >= threshold).astype(int)
    tp = float(((y_true_col == 1) & (y_pred_col == 1)).sum())
    fp = float(((y_true_col == 0) & (y_pred_col == 1)).sum())
    tn = float(((y_true_col == 0) & (y_pred_col == 0)).sum())
    fn = float(((y_true_col == 1) & (y_pred_col == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
    }

def optimize_multilabel_decision_thresholds(y_true, y_prob, args):
    """只使用验证集为每个疾病寻找独立判阳阈值，避免测试集信息泄露。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_labels = y_true.shape[1]

    threshold_min = getattr(args, "threshold_opt_min", 0.10)
    threshold_max = getattr(args, "threshold_opt_max", 0.90)
    threshold_step = getattr(args, "threshold_opt_step", 0.05)
    thresholds_grid = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)

    fallback_threshold = getattr(args, "threshold_opt_fallback", 0.5)
    min_pos = getattr(args, "threshold_opt_min_pos", 2)
    min_neg = getattr(args, "threshold_opt_min_neg", 2)
    min_recall = getattr(args, "threshold_opt_min_recall", 0.0)
    min_specificity = getattr(args, "threshold_opt_min_specificity", 0.0)
    low_prev_threshold = getattr(args, "adaptive_bidir_low_prevalence_threshold", 0.25)
    high_prev_threshold = getattr(args, "adaptive_bidir_high_prevalence_threshold", 0.65)

    best_thresholds = np.full(n_labels, fallback_threshold, dtype=float)
    threshold_info = []

    for label_idx in range(n_labels):
        y_true_col = y_true[:, label_idx]
        y_prob_col = y_prob[:, label_idx]
        pos_count = int(y_true_col.sum())
        neg_count = int(len(y_true_col) - pos_count)
        prevalence = pos_count / max(len(y_true_col), 1)

        # 验证集中某个疾病正样本或负样本太少时，阈值搜索容易过拟合，回退到默认 0.5。
        if pos_count < min_pos or neg_count < min_neg:
            metrics = _binary_metrics_for_threshold(y_true_col, y_prob_col, fallback_threshold)
            threshold_info.append({
                "class_index": label_idx,
                "threshold": fallback_threshold,
                "reason": "fallback_small_val_count",
                "pos_count": pos_count,
                "neg_count": neg_count,
                **metrics,
            })
            continue

        best_score = -float("inf")
        best_metrics = None
        best_reason = "optimized"
        fallback_best_score = -float("inf")
        fallback_best_threshold = fallback_threshold
        fallback_best_metrics = None

        for threshold in thresholds_grid:
            metrics = _binary_metrics_for_threshold(y_true_col, y_prob_col, threshold)

            # 根据疾病阳性占比调整阈值选择倾向：少阳性疾病保护召回，高阳性疾病保护特异度。
            if prevalence <= low_prev_threshold:
                score = metrics["f1"] + 0.45 * metrics["recall"] + 0.20 * metrics["specificity"]
            elif prevalence >= high_prev_threshold:
                score = metrics["f1"] + 0.45 * metrics["specificity"] + 0.20 * metrics["recall"]
            else:
                score = metrics["f1"] + 0.30 * metrics["recall"] + 0.30 * metrics["specificity"]

            if score > fallback_best_score:
                fallback_best_score = score
                fallback_best_threshold = float(threshold)
                fallback_best_metrics = metrics

            if metrics["recall"] < min_recall or metrics["specificity"] < min_specificity:
                continue
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics

        if best_metrics is None:
            best_threshold = fallback_best_threshold
            best_metrics = fallback_best_metrics
            best_reason = "optimized_without_constraints"

        best_thresholds[label_idx] = best_threshold
        threshold_info.append({
            "class_index": label_idx,
            "threshold": best_threshold,
            "reason": best_reason,
            "pos_count": pos_count,
            "neg_count": neg_count,
            **best_metrics,
        })

    print("疾病级阈值优化:", np.round(best_thresholds, 3).tolist())
    return best_thresholds, threshold_info

def optimize_typed_multilabel_decision_thresholds(y_true, y_prob, args):
    """只用验证集进行疾病分型阈值优化，同时约束召回率和特异度。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_labels = y_true.shape[1]

    threshold_min = getattr(args, "threshold_opt_min", 0.10)
    threshold_max = getattr(args, "threshold_opt_max", 0.90)
    threshold_step = getattr(args, "threshold_opt_step", 0.05)
    thresholds_grid = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)

    fallback_threshold = getattr(args, "threshold_opt_fallback", 0.5)
    min_pos = getattr(args, "threshold_opt_min_pos", 2)
    min_neg = getattr(args, "threshold_opt_min_neg", 2)
    base_min_recall = getattr(args, "threshold_opt_min_recall", 0.0)
    base_min_specificity = getattr(args, "threshold_opt_min_specificity", 0.0)
    threshold_feedback = compute_multilabel_feedback(y_true, y_prob, threshold=fallback_threshold)
    disease_types = classify_disease_types(threshold_feedback, args, y_true.mean(axis=0))

    threshold_score_weights = {
        "f1": getattr(args, "threshold_score_f1_weight", 0.40),
        "recall": getattr(args, "threshold_score_recall_weight", 0.25),
        "specificity": getattr(args, "threshold_score_specificity_weight", 0.20),
        "precision": getattr(args, "threshold_score_precision_weight", 0.15),
    }
    threshold_score_weight_sum = sum(threshold_score_weights.values())

    def threshold_composite_score(metrics):
        """计算单个疾病在当前阈值下的综合决策得分。"""
        return sum(
            threshold_score_weights[name] * metrics[name]
            for name in threshold_score_weights
        ) / threshold_score_weight_sum

    best_thresholds = np.full(n_labels, fallback_threshold, dtype=float)
    threshold_info = []

    for label_idx in range(n_labels):
        y_true_col = y_true[:, label_idx]
        y_prob_col = y_prob[:, label_idx]
        pos_count = int(y_true_col.sum())
        neg_count = int(len(y_true_col) - pos_count)
        disease_type = disease_types[label_idx]

        if pos_count < min_pos or neg_count < min_neg:
            metrics = _binary_metrics_for_threshold(y_true_col, y_prob_col, fallback_threshold)
            composite_score = threshold_composite_score(metrics)
            threshold_info.append({
                "class_index": label_idx,
                "threshold": fallback_threshold,
                "reason": "fallback_small_val_count",
                "disease_type": disease_type,
                "constraint_violation": 0.0,
                "composite_score": composite_score,
                "objective_score": composite_score,
                "pos_count": pos_count,
                "neg_count": neg_count,
                **metrics,
            })
            continue

        min_recall = base_min_recall
        min_specificity = base_min_specificity
        if disease_type == "low_prevalence_miss":
            min_recall = max(min_recall, getattr(args, "threshold_typed_low_prev_min_recall", 0.40))
            min_specificity = max(min_specificity, getattr(args, "threshold_typed_low_prev_min_specificity", 0.20))
        elif disease_type == "high_prevalence_false_positive":
            min_recall = max(min_recall, getattr(args, "threshold_typed_high_prev_min_recall", 0.60))
            min_specificity = max(min_specificity, getattr(args, "threshold_typed_high_prev_min_specificity", 0.20))
        elif disease_type == "stable_protected":
            min_recall = max(min_recall, getattr(args, "threshold_typed_stable_min_recall", 0.60))
            min_specificity = max(min_specificity, getattr(args, "threshold_typed_stable_min_specificity", 0.60))
        elif disease_type == "bidirectional_confusion":
            min_recall = max(min_recall, getattr(args, "threshold_typed_confusion_min_recall", 0.35))
            min_specificity = max(min_specificity, getattr(args, "threshold_typed_confusion_min_specificity", 0.35))

        best_threshold = fallback_threshold
        best_metrics = _binary_metrics_for_threshold(y_true_col, y_prob_col, fallback_threshold)
        best_violation = float("inf")
        best_score = -float("inf")

        for threshold in thresholds_grid:
            metrics = _binary_metrics_for_threshold(y_true_col, y_prob_col, threshold)

            # 统一优化多指标综合效果，疾病分型继续负责不同的 Recall/Specificity 约束。
            score = threshold_composite_score(metrics)

            violation = max(min_recall - metrics["recall"], 0.0) + max(min_specificity - metrics["specificity"], 0.0)
            score -= getattr(args, "threshold_typed_constraint_penalty", 2.0) * violation
            score -= getattr(args, "threshold_typed_distance_penalty", 0.05) * abs(float(threshold) - fallback_threshold)

            if (score > best_score + 1e-12) or (
                abs(score - best_score) <= 1e-12 and violation < best_violation
            ):
                best_violation = violation
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics

        best_thresholds[label_idx] = best_threshold
        threshold_info.append({
            "class_index": label_idx,
            "threshold": best_threshold,
            "reason": "optimized" if best_violation <= 0 else "optimized_with_soft_constraints",
            "disease_type": disease_type,
            "constraint_violation": best_violation,
            "composite_score": threshold_composite_score(best_metrics),
            "objective_score": best_score,
            "pos_count": pos_count,
            "neg_count": neg_count,
            **best_metrics,
        })

    print("疾病分型阈值优化:", np.round(best_thresholds, 3).tolist())
    return best_thresholds, threshold_info

def update_negative_sampling_alpha(alpha, feedback, args):
    """根据验证集反馈更新每个疾病标签的难阴性采样强度。

    只有当某个标签假阳性率偏高且召回率仍可接受时，才提高 alpha[c]。
    如果召回率偏低或假阴性率偏高，则降低 alpha[c]，避免模型过度偏向阴性预测。
    """
    high_fp = (feedback["fpr"] > args.adaptive_neg_tau_fp) & (
        feedback["recall"] >= args.adaptive_neg_tau_recall
    )
    high_fn = (feedback["fnr"] > args.adaptive_neg_tau_fn) | (
        feedback["recall"] < args.adaptive_neg_tau_recall
    )

    updated = alpha.copy()
    updated[high_fp] *= args.adaptive_neg_growth
    updated[high_fn] *= args.adaptive_neg_decay
    updated = np.clip(updated, args.adaptive_neg_alpha_min, args.adaptive_neg_alpha_max)

    return updated

def update_bidirectional_sampling_alpha(alpha_pos, alpha_neg, feedback, args, prevalence=None, train_feedback=None):
    """根据验证集反馈平衡更新每个疾病的阳性和阴性采样强度。

    这里不用简单的“超过阈值就翻倍”，而是把 FNR/Recall 看作阳性压力，
    把 FPR/Specificity 看作阴性压力。哪边压力更大，就更温和地增强哪边；
    如果某边在训练集明显好于验证集，则降低该方向采样强度，减少过拟合。
    """
    has_pos = feedback["pos_count"] > 0
    has_neg = feedback["neg_count"] > 0
    if prevalence is None:
        prevalence = np.full_like(feedback["recall"], 0.5, dtype=float)
    else:
        prevalence = np.asarray(prevalence, dtype=float)

    # 连续压力分数：越超过阈值，说明该方向越需要增强。
    pos_pressure = (
        np.maximum(feedback["fnr"] - args.adaptive_bidir_tau_fnr, 0.0)
        + np.maximum(args.adaptive_bidir_tau_recall - feedback["recall"], 0.0)
    ) * has_pos.astype(float)
    neg_pressure = (
        np.maximum(feedback["fpr"] - args.adaptive_bidir_tau_fpr, 0.0)
        + np.maximum(args.adaptive_bidir_tau_specificity - feedback["specificity"], 0.0)
    ) * has_neg.astype(float)

    use_prevalence_prior = not getattr(args, "disable_adaptive_bidir_prevalence_prior", False)
    low_prevalence = prevalence <= getattr(args, "adaptive_bidir_low_prevalence_threshold", 0.25)
    high_prevalence = prevalence >= getattr(args, "adaptive_bidir_high_prevalence_threshold", 0.65)
    prevalence_boost = getattr(args, "adaptive_bidir_prevalence_boost", 1.15)
    opposite_scale = getattr(args, "adaptive_bidir_opposite_prevalence_scale", 0.75)

    if use_prevalence_prior:
        # 先验只做轻量修正，避免上一版那样把模型整体推成阳性预测偏置。
        pos_pressure[low_prevalence] *= prevalence_boost
        neg_pressure[high_prevalence] *= prevalence_boost
        pos_pressure[high_prevalence] *= opposite_scale
        neg_pressure[low_prevalence] *= opposite_scale

    step = max(args.adaptive_bidir_growth - 1.0, 0.0)
    pressure_scale = getattr(args, "adaptive_bidir_pressure_scale", 1.0)
    pos_factor = 1.0 + step * np.clip(pos_pressure * pressure_scale, 0.0, 1.0)
    neg_factor = 1.0 + step * np.clip(neg_pressure * pressure_scale, 0.0, 1.0)

    updated_pos = alpha_pos.copy() * pos_factor
    updated_neg = alpha_neg.copy() * neg_factor

    # 如果一侧错误明显高于另一侧，就衰减相反方向，避免阳性/阴性同时越采越强。
    balance_margin = getattr(args, "adaptive_bidir_balance_margin", 0.15)
    pos_dominant = pos_pressure > (neg_pressure + balance_margin)
    neg_dominant = neg_pressure > (pos_pressure + balance_margin)
    updated_neg[pos_dominant] *= args.adaptive_bidir_decay
    updated_pos[neg_dominant] *= args.adaptive_bidir_decay

    # 护栏：FPR 或 FNR 极高时强制纠偏，防止出现全阳性或全阴性的退化模式。
    high_fpr_guard = feedback["fpr"] >= getattr(args, "adaptive_bidir_high_fpr_guard", 0.75)
    high_fnr_guard = feedback["fnr"] >= getattr(args, "adaptive_bidir_high_fnr_guard", 0.75)
    guard_growth = getattr(args, "adaptive_bidir_guard_growth", 1.15)
    guard_decay = getattr(args, "adaptive_bidir_guard_decay", 0.8)
    updated_neg[high_fpr_guard] *= guard_growth
    updated_pos[high_fpr_guard] *= guard_decay
    updated_pos[high_fnr_guard] *= guard_growth
    updated_neg[high_fnr_guard] *= guard_decay

    # 过拟合保护：训练集某方向明显好于验证集时，不继续强化该方向。
    if train_feedback is not None:
        overfit_gap = getattr(args, "adaptive_bidir_overfit_gap", 0.25)
        overfit_decay = getattr(args, "adaptive_bidir_overfit_decay", 0.85)
        pos_overfit = (train_feedback["recall"] - feedback["recall"]) > overfit_gap
        neg_overfit = (train_feedback["specificity"] - feedback["specificity"]) > overfit_gap
        updated_pos[pos_overfit] *= overfit_decay
        updated_neg[neg_overfit] *= overfit_decay

    # 压力很低的标签逐步回到 1，降低长期重复采样导致的方差和过拟合。
    stable = (pos_pressure < 1e-8) & (neg_pressure < 1e-8) & ~high_fpr_guard & ~high_fnr_guard
    updated_pos[stable] = 1.0 + (updated_pos[stable] - 1.0) * args.adaptive_bidir_decay
    updated_neg[stable] = 1.0 + (updated_neg[stable] - 1.0) * args.adaptive_bidir_decay

    pos_alpha_max = getattr(args, "adaptive_bidir_pos_alpha_max", args.adaptive_bidir_alpha_max)
    neg_alpha_max = getattr(args, "adaptive_bidir_neg_alpha_max", args.adaptive_bidir_alpha_max)
    updated_pos = np.clip(updated_pos, args.adaptive_bidir_alpha_min, pos_alpha_max)
    updated_neg = np.clip(updated_neg, args.adaptive_bidir_alpha_min, neg_alpha_max)

    print(
        "双向采样反馈: "
        f"pos_pressure={pos_pressure.mean():.3f}, neg_pressure={neg_pressure.mean():.3f}, "
        f"alpha_pos={updated_pos.mean():.3f}, alpha_neg={updated_neg.mean():.3f}"
    )

    return updated_pos, updated_neg

def build_disease_typed_bidir_params(feedback, args, prevalence):
    """根据疾病错误类型，为每个疾病生成不同的双向采样参数。"""
    prevalence = np.asarray(prevalence, dtype=float)
    n_labels = len(prevalence)

    params = {
        "tau_fnr": np.full(n_labels, args.adaptive_bidir_tau_fnr, dtype=float),
        "tau_fpr": np.full(n_labels, args.adaptive_bidir_tau_fpr, dtype=float),
        "tau_recall": np.full(n_labels, args.adaptive_bidir_tau_recall, dtype=float),
        "tau_specificity": np.full(n_labels, args.adaptive_bidir_tau_specificity, dtype=float),
        "pos_alpha_max": np.full(n_labels, getattr(args, "adaptive_bidir_pos_alpha_max", args.adaptive_bidir_alpha_max), dtype=float),
        "neg_alpha_max": np.full(n_labels, getattr(args, "adaptive_bidir_neg_alpha_max", args.adaptive_bidir_alpha_max), dtype=float),
        "growth_pos": np.full(n_labels, args.adaptive_bidir_growth, dtype=float),
        "growth_neg": np.full(n_labels, args.adaptive_bidir_growth, dtype=float),
        "decay_pos": np.full(n_labels, args.adaptive_bidir_decay, dtype=float),
        "decay_neg": np.full(n_labels, args.adaptive_bidir_decay, dtype=float),
        "pos_hard_threshold": np.full(n_labels, args.adaptive_bidir_pos_hard_threshold, dtype=float),
        "pos_boundary_threshold": np.full(n_labels, args.adaptive_bidir_pos_boundary_threshold, dtype=float),
        "neg_boundary_threshold": np.full(n_labels, args.adaptive_bidir_neg_boundary_threshold, dtype=float),
        "neg_hard_threshold": np.full(n_labels, args.adaptive_bidir_neg_hard_threshold, dtype=float),
        "pos_boundary_bonus": np.full(n_labels, args.adaptive_bidir_pos_boundary_bonus, dtype=float),
        "pos_hard_bonus": np.full(n_labels, args.adaptive_bidir_pos_hard_bonus, dtype=float),
        "neg_boundary_bonus": np.full(n_labels, args.adaptive_bidir_neg_boundary_bonus, dtype=float),
        "neg_hard_bonus": np.full(n_labels, args.adaptive_bidir_neg_hard_bonus, dtype=float),
    }

    recall = feedback["recall"]
    fnr = feedback["fnr"]
    specificity = feedback["specificity"]
    fpr = feedback["fpr"]

    very_low_prev = prevalence <= getattr(args, "disease_typed_very_low_prevalence_threshold", 0.10)
    low_prev = prevalence <= getattr(args, "adaptive_bidir_low_prevalence_threshold", 0.25)
    high_prev = prevalence >= getattr(args, "adaptive_bidir_high_prevalence_threshold", 0.65)
    very_high_prev = prevalence >= getattr(args, "disease_typed_very_high_prevalence_threshold", 0.90)

    miss_problem = (fnr >= args.adaptive_bidir_tau_fnr) | (recall <= args.adaptive_bidir_tau_recall)
    fp_problem = (fpr >= args.adaptive_bidir_tau_fpr) | (specificity <= args.adaptive_bidir_tau_specificity)
    disease_type = classify_disease_types(feedback, args, prevalence)

    # 少阳性漏诊型：更保护阳性，抑制阴性增强，避免少数阳性疾病被压成全阴性。
    # 稳定疾病保护：验证集已经较平衡时，关闭额外采样，避免把已学好的疾病越调越坏。
    mask = disease_type == "stable_protected"
    params["tau_fnr"][mask] = 1.0
    params["tau_fpr"][mask] = 1.0
    params["tau_recall"][mask] = 0.0
    params["tau_specificity"][mask] = 0.0
    params["pos_alpha_max"][mask] = 1.0
    params["neg_alpha_max"][mask] = 1.0
    params["growth_pos"][mask] = 1.0
    params["growth_neg"][mask] = 1.0
    params["decay_pos"][mask] = getattr(args, "disease_typed_stable_decay", 0.50)
    params["decay_neg"][mask] = getattr(args, "disease_typed_stable_decay", 0.50)
    params["pos_boundary_bonus"][mask] = 0.0
    params["pos_hard_bonus"][mask] = 0.0
    params["neg_boundary_bonus"][mask] = 0.0
    params["neg_hard_bonus"][mask] = 0.0

    mask = disease_type == "low_prevalence_miss"
    params["tau_fnr"][mask] *= 0.85
    params["tau_recall"][mask] += 0.08
    params["pos_alpha_max"][mask] = np.maximum(params["pos_alpha_max"][mask], getattr(args, "disease_typed_low_prev_pos_alpha_max", 3.5))
    params["neg_alpha_max"][mask] = np.minimum(params["neg_alpha_max"][mask], getattr(args, "disease_typed_low_prev_neg_alpha_max", 1.4))
    params["growth_pos"][mask] *= getattr(args, "disease_typed_pos_growth_boost", 1.20)
    params["growth_neg"][mask] = 1.0 + (params["growth_neg"][mask] - 1.0) * getattr(args, "disease_typed_low_prev_neg_growth_scale", 0.45)
    params["neg_boundary_bonus"][mask] *= getattr(args, "disease_typed_low_prev_neg_bonus_scale", 0.50)
    params["neg_hard_bonus"][mask] *= getattr(args, "disease_typed_low_prev_neg_bonus_scale", 0.50)

    # 极少阳性疾病再额外降低阴性干预，优先保证能学到阳性模式。
    mask = very_low_prev & miss_problem
    params["neg_alpha_max"][mask] = np.minimum(params["neg_alpha_max"][mask], getattr(args, "disease_typed_very_low_prev_neg_alpha_max", 1.2))
    params["pos_alpha_max"][mask] = np.maximum(params["pos_alpha_max"][mask], getattr(args, "disease_typed_very_low_prev_pos_alpha_max", 4.0))

    # 高阳性误报型：更关注少量阴性样本，但限制上限，避免对极少阴性过拟合。
    mask = disease_type == "high_prevalence_false_positive"
    params["tau_fpr"][mask] *= 0.85
    params["tau_specificity"][mask] += 0.08
    params["neg_alpha_max"][mask] = np.maximum(params["neg_alpha_max"][mask], getattr(args, "disease_typed_high_prev_neg_alpha_max", 3.2))
    params["pos_alpha_max"][mask] = np.minimum(params["pos_alpha_max"][mask], getattr(args, "disease_typed_high_prev_pos_alpha_max", 1.6))
    params["growth_neg"][mask] *= getattr(args, "disease_typed_neg_growth_boost", 1.20)
    params["growth_pos"][mask] = 1.0 + (params["growth_pos"][mask] - 1.0) * getattr(args, "disease_typed_high_prev_pos_growth_scale", 0.55)
    params["neg_boundary_bonus"][mask] *= getattr(args, "disease_typed_high_prev_neg_bonus_boost", 1.20)
    params["neg_hard_bonus"][mask] *= getattr(args, "disease_typed_high_prev_neg_bonus_boost", 1.20)

    # 极高阳性疾病的阴性样本非常少，增强阴性时保持上限，避免反复记住少量阴性。
    mask = very_high_prev & fp_problem
    params["neg_alpha_max"][mask] = np.minimum(params["neg_alpha_max"][mask], getattr(args, "disease_typed_very_high_prev_neg_alpha_cap", 2.6))

    # 阴性偏置型：增强阳性，轻度抑制阴性。
    mask = disease_type == "negative_bias"
    params["growth_pos"][mask] *= getattr(args, "disease_typed_pos_growth_boost", 1.20)
    params["growth_neg"][mask] = 1.0 + (params["growth_neg"][mask] - 1.0) * 0.60
    params["neg_alpha_max"][mask] = np.minimum(params["neg_alpha_max"][mask], 1.8)

    # 阳性偏置型：增强阴性，轻度抑制阳性。
    mask = disease_type == "positive_bias"
    params["growth_neg"][mask] *= getattr(args, "disease_typed_neg_growth_boost", 1.20)
    params["growth_pos"][mask] = 1.0 + (params["growth_pos"][mask] - 1.0) * 0.60
    params["pos_alpha_max"][mask] = np.minimum(params["pos_alpha_max"][mask], 1.8)

    # 双向混淆型：两侧都增强，但降低单侧过强的风险。
    mask = disease_type == "bidirectional_confusion"
    params["pos_alpha_max"][mask] = np.minimum(params["pos_alpha_max"][mask], getattr(args, "disease_typed_confusion_alpha_max", 2.2))
    params["neg_alpha_max"][mask] = np.minimum(params["neg_alpha_max"][mask], getattr(args, "disease_typed_confusion_alpha_max", 2.2))

    return disease_type, params

def update_disease_typed_bidirectional_sampling_alpha(alpha_pos, alpha_neg, feedback, args, prevalence, train_feedback=None):
    """疾病分型驱动的双向采样强度更新。"""
    disease_type, params = build_disease_typed_bidir_params(feedback, args, prevalence)
    has_pos = feedback["pos_count"] > 0
    has_neg = feedback["neg_count"] > 0

    pos_pressure = (
        np.maximum(feedback["fnr"] - params["tau_fnr"], 0.0)
        + np.maximum(params["tau_recall"] - feedback["recall"], 0.0)
    ) * has_pos.astype(float)
    neg_pressure = (
        np.maximum(feedback["fpr"] - params["tau_fpr"], 0.0)
        + np.maximum(params["tau_specificity"] - feedback["specificity"], 0.0)
    ) * has_neg.astype(float)

    pressure_scale = getattr(args, "adaptive_bidir_pressure_scale", 1.0)
    pos_factor = 1.0 + np.maximum(params["growth_pos"] - 1.0, 0.0) * np.clip(pos_pressure * pressure_scale, 0.0, 1.0)
    neg_factor = 1.0 + np.maximum(params["growth_neg"] - 1.0, 0.0) * np.clip(neg_pressure * pressure_scale, 0.0, 1.0)

    updated_pos = alpha_pos.copy() * pos_factor
    updated_neg = alpha_neg.copy() * neg_factor

    balance_margin = getattr(args, "adaptive_bidir_balance_margin", 0.15)
    pos_dominant = pos_pressure > (neg_pressure + balance_margin)
    neg_dominant = neg_pressure > (pos_pressure + balance_margin)
    updated_neg[pos_dominant] *= params["decay_neg"][pos_dominant]
    updated_pos[neg_dominant] *= params["decay_pos"][neg_dominant]

    high_fpr_guard = feedback["fpr"] >= getattr(args, "adaptive_bidir_high_fpr_guard", 0.75)
    high_fnr_guard = feedback["fnr"] >= getattr(args, "adaptive_bidir_high_fnr_guard", 0.75)
    guard_growth = getattr(args, "adaptive_bidir_guard_growth", 1.15)
    guard_decay = getattr(args, "adaptive_bidir_guard_decay", 0.8)
    updated_neg[high_fpr_guard] *= guard_growth
    updated_pos[high_fpr_guard] *= guard_decay
    updated_pos[high_fnr_guard] *= guard_growth
    updated_neg[high_fnr_guard] *= guard_decay

    if train_feedback is not None:
        overfit_gap = getattr(args, "adaptive_bidir_overfit_gap", 0.25)
        overfit_decay = getattr(args, "adaptive_bidir_overfit_decay", 0.85)
        pos_overfit = (train_feedback["recall"] - feedback["recall"]) > overfit_gap
        neg_overfit = (train_feedback["specificity"] - feedback["specificity"]) > overfit_gap
        updated_pos[pos_overfit] *= overfit_decay
        updated_neg[neg_overfit] *= overfit_decay

    stable = np.isin(disease_type, ["stable", "stable_protected"]) & (pos_pressure < 1e-8) & (neg_pressure < 1e-8) & ~high_fpr_guard & ~high_fnr_guard
    updated_pos[stable] = 1.0 + (updated_pos[stable] - 1.0) * params["decay_pos"][stable]
    updated_neg[stable] = 1.0 + (updated_neg[stable] - 1.0) * params["decay_neg"][stable]

    updated_pos = np.clip(updated_pos, args.adaptive_bidir_alpha_min, params["pos_alpha_max"])
    updated_neg = np.clip(updated_neg, args.adaptive_bidir_alpha_min, params["neg_alpha_max"])

    type_summary = {name: int((disease_type == name).sum()) for name in np.unique(disease_type)}
    print(
        "疾病分型双向采样反馈: "
        f"type={type_summary}, pos_pressure={pos_pressure.mean():.3f}, neg_pressure={neg_pressure.mean():.3f}, "
        f"alpha_pos={updated_pos.mean():.3f}, alpha_neg={updated_neg.mean():.3f}"
    )

    return updated_pos, updated_neg, disease_type, params

@torch.no_grad()
def collect_multilabel_predictions(model, device, loader, sourceint, source2task):
    """按 DataLoader 顺序收集多标签预测概率。

    训练集反馈使用不打乱顺序的 loader，确保返回数组与 WeightedRandomSampler
    所需的 dataset 索引顺序保持一致。
    """
    model.eval()

    y_prob_list = []
    y_true_list = []
    task_type = get_task(sourceint, source2task)

    for x, y, _, _, _ in loader:
        x = x.to(device)
        pred = model(x, sourceint)

        if getattr(model, 'use_rule_mask', False):
            pred = apply_hard_rules(x, pred, task_type, getattr(model, 'feat_indices', {}))

        y_prob_list.append(torch.sigmoid(pred).cpu())
        y_true_list.append(y.cpu())

    y_prob = torch.cat(y_prob_list, dim=0).numpy()
    y_true = torch.cat(y_true_list, dim=0).numpy()

    return y_true, y_prob

def build_adaptive_negative_loader(dataset, batch_size, y_true, y_prob, alpha, args):
    """构建下一轮训练使用的难阴性增强 DataLoader。

    一个样本可能同时是多个疾病标签的阴性样本。这里取最大的标签级 bonus
    作为样本权重增量，避免同一患者在多个标签上都是难阴性时权重被过度放大。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    # 难阴性样本是潜在假阳性病例；边界阴性样本靠近决策阈值，用于细化分类边界。
    negative_mask = y_true == 0
    hard_neg = negative_mask & (y_prob >= args.adaptive_neg_hard_threshold)
    boundary_neg = negative_mask & (
        (y_prob >= args.adaptive_neg_boundary_threshold)
        & (y_prob < args.adaptive_neg_hard_threshold)
    )

    class_bonus = (
        hard_neg.astype(float) * args.adaptive_neg_hard_bonus
        + boundary_neg.astype(float) * args.adaptive_neg_boundary_bonus
    ) * alpha.reshape(1, -1)

    sample_bonus = class_bonus.max(axis=1)
    sample_weights = 1.0 + sample_bonus
    # 将平均权重归一到 1 附近，让每轮采样数量而不是原始权重尺度控制增强强度。
    sample_weights = sample_weights / max(sample_weights.mean(), 1e-8)

    num_samples = max(1, int(round(len(dataset) * args.adaptive_neg_epoch_multiplier)))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )

    hard_count = hard_neg.any(axis=1).sum()
    boundary_count = boundary_neg.any(axis=1).sum()
    print(
        "自适应难阴性采样: "
        f"hard={int(hard_count)}, boundary={int(boundary_count)}, "
        f"alpha_mean={alpha.mean():.3f}, weight_max={sample_weights.max():.3f}"
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

def build_bidirectional_adaptive_loader(dataset, batch_size, y_true, y_prob, alpha_pos, alpha_neg, args):
    """构建双向自适应采样 DataLoader。

    阳性漏诊时增加难阳性样本权重，阴性误报时增加难阴性样本权重。
    由于一个患者同时对应 8 个疾病标签，这里先在标签维度计算 bonus，再取最大值作为患者级采样权重，
    避免一个患者因为多个标签同时被增强而权重过度膨胀。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    positive_mask = y_true == 1
    negative_mask = y_true == 0

    # 难阳性：真实阳性但预测概率低，主要对应漏诊问题。
    hard_pos = positive_mask & (y_prob < args.adaptive_bidir_pos_hard_threshold)
    boundary_pos = positive_mask & (
        (y_prob >= args.adaptive_bidir_pos_hard_threshold)
        & (y_prob < args.adaptive_bidir_pos_boundary_threshold)
    )

    # 难阴性：真实阴性但预测概率高，主要对应误报问题。
    hard_neg = negative_mask & (y_prob >= args.adaptive_bidir_neg_hard_threshold)
    boundary_neg = negative_mask & (
        (y_prob >= args.adaptive_bidir_neg_boundary_threshold)
        & (y_prob < args.adaptive_bidir_neg_hard_threshold)
    )

    # alpha=1 表示不额外增强；只有反馈把某个方向推到 1 以上时，才真正增加采样概率。
    # 这样可以避免训练初期大量阴性标签天然主导双向采样。
    alpha_pos_effective = np.maximum(alpha_pos - 1.0, 0.0)
    alpha_neg_effective = np.maximum(alpha_neg - 1.0, 0.0)

    pos_bonus = (
        hard_pos.astype(float) * args.adaptive_bidir_pos_hard_bonus
        + boundary_pos.astype(float) * args.adaptive_bidir_pos_boundary_bonus
    ) * alpha_pos_effective.reshape(1, -1)
    neg_bonus = (
        hard_neg.astype(float) * args.adaptive_bidir_neg_hard_bonus
        + boundary_neg.astype(float) * args.adaptive_bidir_neg_boundary_bonus
    ) * alpha_neg_effective.reshape(1, -1)

    # 分别取阳性方向和阴性方向的最大 bonus，再相加；并设置上限，避免少数患者反复被过采样。
    sample_bonus = pos_bonus.max(axis=1) + neg_bonus.max(axis=1)
    sample_bonus = np.clip(
        sample_bonus,
        0.0,
        getattr(args, "adaptive_bidir_sample_bonus_max", 3.0)
    )
    sample_weights = 1.0 + sample_bonus
    sample_weights = sample_weights / max(sample_weights.mean(), 1e-8)

    num_samples = max(1, int(round(len(dataset) * args.adaptive_bidir_epoch_multiplier)))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )

    hard_pos_count = hard_pos.any(axis=1).sum()
    hard_neg_count = hard_neg.any(axis=1).sum()
    print(
        "双向自适应采样: "
        f"hard_pos={int(hard_pos_count)}, hard_neg={int(hard_neg_count)}, "
        f"alpha_pos_eff={alpha_pos_effective.mean():.3f}, alpha_neg_eff={alpha_neg_effective.mean():.3f}, "
        f"weight_max={sample_weights.max():.3f}"
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

def build_disease_typed_bidirectional_loader(dataset, batch_size, y_true, y_prob, alpha_pos, alpha_neg, params, args):
    """使用疾病级阈值和 bonus 构建双向采样 DataLoader。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    positive_mask = y_true == 1
    negative_mask = y_true == 0

    pos_hard_threshold = params["pos_hard_threshold"].reshape(1, -1)
    pos_boundary_threshold = params["pos_boundary_threshold"].reshape(1, -1)
    neg_boundary_threshold = params["neg_boundary_threshold"].reshape(1, -1)
    neg_hard_threshold = params["neg_hard_threshold"].reshape(1, -1)

    hard_pos = positive_mask & (y_prob < pos_hard_threshold)
    boundary_pos = positive_mask & (
        (y_prob >= pos_hard_threshold)
        & (y_prob < pos_boundary_threshold)
    )
    hard_neg = negative_mask & (y_prob >= neg_hard_threshold)
    boundary_neg = negative_mask & (
        (y_prob >= neg_boundary_threshold)
        & (y_prob < neg_hard_threshold)
    )

    alpha_pos_effective = np.maximum(alpha_pos - 1.0, 0.0)
    alpha_neg_effective = np.maximum(alpha_neg - 1.0, 0.0)

    pos_bonus = (
        hard_pos.astype(float) * params["pos_hard_bonus"].reshape(1, -1)
        + boundary_pos.astype(float) * params["pos_boundary_bonus"].reshape(1, -1)
    ) * alpha_pos_effective.reshape(1, -1)
    neg_bonus = (
        hard_neg.astype(float) * params["neg_hard_bonus"].reshape(1, -1)
        + boundary_neg.astype(float) * params["neg_boundary_bonus"].reshape(1, -1)
    ) * alpha_neg_effective.reshape(1, -1)

    sample_bonus = pos_bonus.max(axis=1) + neg_bonus.max(axis=1)
    sample_bonus = np.clip(
        sample_bonus,
        0.0,
        getattr(args, "adaptive_bidir_sample_bonus_max", 3.0)
    )
    sample_weights = 1.0 + sample_bonus
    sample_weights = sample_weights / max(sample_weights.mean(), 1e-8)

    num_samples = max(1, int(round(len(dataset) * args.adaptive_bidir_epoch_multiplier)))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )

    print(
        "疾病分型双向采样器: "
        f"hard_pos={int(hard_pos.any(axis=1).sum())}, hard_neg={int(hard_neg.any(axis=1).sum())}, "
        f"alpha_pos_eff={alpha_pos_effective.mean():.3f}, alpha_neg_eff={alpha_neg_effective.mean():.3f}, "
        f"weight_max={sample_weights.max():.3f}"
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

def train(model, device, train_loader, optimizer):
    # print("---Train---")
    model.train()

    for x, y, sourceint_list, __, __ in train_loader:
        # x (batch size, number of dimensions)
        # y (batch size)
        # sourceint_list (batch_size)
        optimizer.zero_grad()
        x, y, source = x.to(device), y.to(device), sourceint_list.to(device)
        batch_loss = model.compute_loss(x, y, sourceint_list)
        batch_loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(model, device, loader, sourceint, source2task, writer, epoch, decision_thresholds=None):
    # A single source is passed in here with a corresponding loader object
    model.eval()

    # Prepare the vectors corresponding to the model predictions
    y_pred_list, y_prob_list, y_true_list = [], [], []
    drug_list, entity_list, source_list = [], [], []

    # 获取当前任务类型
    current_task = get_task(sourceint, source2task)

    for x, y, loader_sourceint_list, entity_idx, drug_idxs in loader:
        assert torch.all(loader_sourceint_list == sourceint)
        x = x.to(device)

        pred = model(x, sourceint)


        # assert(get_task(sourceint, source2task) == "numeric")
        # y_pred_list.append(pred.view(-1,))
        # y_prob_list.append(pred.view(-1))

        # === 修改逻辑开始 ===
        if current_task == "numeric":
            y_pred_list.append(pred.view(-1,))
            y_prob_list.append(pred.view(-1))
        elif current_task == "multilabel":
            # ====== 新增：在评估时，进行医学红线拦截 ======
            if getattr(model, 'use_rule_mask', False):
                pred = apply_hard_rules(x, pred, current_task, getattr(model, 'feat_indices', {}))
            # ==============================================

            # 多标签分类不需要 view(-1)，保持 (Batch, Classes) 形状
            y_pred_list.append(pred)
            y_prob_list.append(torch.sigmoid(pred)) # 保存概率用于查看
        else:
            raise ValueError(f"Unknown task: {current_task}")
        # === 修改逻辑结束 ===

        y_true_list.append(y)
        drug_list.append(drug_idxs)
        entity_list.append(entity_idx)
        source_list.append(loader_sourceint_list)

    y_pred = torch.cat(y_pred_list, dim = 0).cpu().detach()
    y_prob = torch.cat(y_prob_list, dim = 0).cpu().detach() # predictions for regression
    y_true = torch.cat(y_true_list, dim = 0)
    drug_list = torch.cat(drug_list, dim = 0)
    entity_list = torch.cat(entity_list, dim = 0)
    source_list = torch.cat(source_list, dim = 0)

    # # Compute the corresponding eval_dict
    # assert(get_task(sourceint, source2task) == "numeric")
    # # dt(y_true, "y_true")
    # # dt(y_pred, "y_pred")
    # metric2score, _ = RegEval().evaluate_all(y_true.numpy(), y_pred.numpy()) # Doublecheck correctness for regression

    # # === 修改评估调用 ===
    # if current_task == "numeric":
    #     metric2score, _ = RegEval().evaluate_all(y_true.numpy(), y_pred.numpy())
    # elif current_task == "multilabel":
    #     # 注意：MultilabelEval 内部会处理 logits，所以传入 y_pred (logits)
    #     metric2score, _ = MultilabelEval().evaluate_all(y_true.numpy(), y_pred.numpy())
    # # ====================
    # === 修改评估调用，捕获 patient_scores ===
    patient_scores = {}
    if current_task == "numeric":
        metric2score, patient_scores = RegEval().evaluate_all(y_true.numpy(), y_pred.numpy())
    elif current_task == "multilabel":
        metric2score, patient_scores = MultilabelEval(decision_thresholds=decision_thresholds).evaluate_all(
            y_true.numpy(), y_pred.numpy()
        )
    # ==========================================

    # Additionally compute the negative of the loss (positive means better for all metrics in metric2score)
    metric2score["neg_loss"] = 0
    n_samples = 0
    for x, y, sourceint_list, _, _ in loader:
        # x (batch size, number of dimensions)
        # y (batch size)
        # sourceint_list (batch_size)
        # Calculate batch loss
        x, y, source = x.to(device), y.to(device), sourceint_list.to(device)
        n_samples_in_batch = x.shape[0]
        batch_neg_loss_sum = n_samples_in_batch*model.compute_loss(x, y, sourceint_list)

        # Update
        metric2score["neg_loss"] -= batch_neg_loss_sum
        n_samples += n_samples_in_batch

    metric2score["neg_loss"] /= float(n_samples)
    metric2score["neg_loss"] = metric2score["neg_loss"].cpu().detach().item()

    # Add to writer
    for metric, score in metric2score.items():
        if "neg_" in metric:
            metric = metric.split("neg_")[1]
            score = -1*score
        writer.add_scalar(f'{metric}', score, epoch)

    # return metric2score, y_prob, y_true, drug_list, entity_list, source_list
    # 返回值增加 patient_scores
    return metric2score, y_prob, y_true, drug_list, entity_list, source_list, patient_scores

def set_drug_representation(provider, drugkg):
    if drugkg:
        print('Use KG embedding to represent drugs')
        # Sum pool over embeddings of drugs that are used
        provider.y_dict['drug'] = (provider.y_dict['drug'].to(torch.float32)).matmul(provider.kg.data.x[provider.kg_mapping_dict['drug']])

    return provider

def subsample_dataset(dataset, sample_frac):
    indices = torch.randint(low = 0, high = len(dataset), size = (int(sample_frac*len(dataset)),))
    dataset = torch.utils.data.Subset(dataset, indices)
    return dataset, indices

def get_split_dataset_and_loader(provider, sourceint_list, split_idx, batch_size, shuffle, sample_frac):
    split_dataset = provider.get_dataset(idx_list = split_idx, sourceint = sourceint_list)
    entity_ids = split_dataset.entity_ids
    drug_names = split_dataset.drug_names

    if sample_frac < 1:
        print("WARNING: sample_frac < 1 only applied to finetune dataset")
        split_dataset, indices = subsample_dataset(split_dataset, sample_frac)
        split_dataset.entity_ids = entity_ids # BUG: entity_ids not necessary correct for split_dataset; need to review
        split_dataset.drug_names = drug_names # BUG: drug_names not necessary correct for split_dataset; need to review

    split_loader = DataLoader(split_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0)

    return split_dataset, split_loader

def get_sourceint_list(training_type, finetune_sourceint_list, pretrain_sourceint_list):
    # Get sourceint
    if training_type == "finetune":
        sourceint_list = finetune_sourceint_list
    elif training_type == "pretrain":
        sourceint_list = pretrain_sourceint_list
    else:
        assert(False)

    return sourceint_list

def get_shuffle(split):
    # Get shuffle
    if split == "train":
        shuffle = True
    else:
        shuffle = False

    return shuffle

def _provider_with_foldwise_minmax(provider, train_idx):
    """Return a shallow provider copy whose features are scaled from train only.

    ``train_idx`` indexes rows in ``y_dict``.  The corresponding unique patient
    entities are therefore recovered before estimating column-wise extrema.
    Validation and test patients never contribute to the fitted transform.
    """
    if not hasattr(provider, "X_expression") or not hasattr(provider, "y_dict"):
        raise AttributeError("fold-wise min-max scaling requires X_expression and y_dict")

    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    train_entity_idx = torch.unique(provider.y_dict["entity"][train_idx])
    if train_entity_idx.numel() == 0:
        raise ValueError("cannot fit fold-wise min-max scaling on an empty training split")

    all_features = provider.X_expression.to(torch.float32)
    train_features = all_features[train_entity_idx]
    feature_min = train_features.amin(dim=0)
    feature_max = train_features.amax(dim=0)
    feature_range = feature_max - feature_min
    nonconstant = feature_range > 0

    scaled_features = torch.zeros_like(all_features, dtype=torch.float32)
    scaled_features[:, nonconstant] = (
        all_features[:, nonconstant] - feature_min[nonconstant]
    ) / feature_range[nonconstant]

    fold_provider = copy.copy(provider)
    fold_provider.X_expression = scaled_features
    fold_provider.foldwise_minmax_params = {
        "feature_min": feature_min,
        "feature_max": feature_max,
        "train_entity_idx": train_entity_idx,
    }
    return fold_provider


def get_training_type2split2source2dataset_or_loader(provider, finetune_sourceint_list, pretrain_sourceint_list, train_idx, val_idx, test_idx, batch_size, sample_frac, foldwise_minmax=False):
    if foldwise_minmax:
        provider = _provider_with_foldwise_minmax(provider, train_idx)

    # Initialize
    training_type2split2source2dataset_or_loader = {"finetune": {"train": dict(), "val": dict(), "test": dict()}, "pretrain": {"train": dict(), "val": dict(), "test": dict()}}
    
    for training_type in ["finetune", "pretrain"]:
        sample_frac_to_use = {"finetune": sample_frac, "pretrain": 1}.get(training_type)

        # Get sourceint_list
        sourceint_list = get_sourceint_list(training_type, finetune_sourceint_list, pretrain_sourceint_list)

        # Populate by split
        for split, split_idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            # Get shuffle
            shuffle = get_shuffle(split)

            # Iterate over sourceint list
            for sourceint_i in sourceint_list:
                dataset_i, loader_i = get_split_dataset_and_loader(provider, sourceint_list = [sourceint_i], split_idx = split_idx, batch_size = batch_size, shuffle = shuffle, sample_frac = sample_frac_to_use)

                training_type2split2source2dataset_or_loader[training_type][split][sourceint_i] = {"dataset": dataset_i, "loader": loader_i}

    return training_type2split2source2dataset_or_loader

def print_dataset_sizes(training_type2split2source2dataset_or_loader, pretrain_sourceint_list, finetune_sourceint_list):
    for training_type in ["finetune"]: # ["pretrain", "finetune"]
        sourceint_list = get_sourceint_list(training_type, finetune_sourceint_list, pretrain_sourceint_list)
        for source in sourceint_list:
            dataset_split_size_list = np.array([len(training_type2split2source2dataset_or_loader[training_type]["train"][source]["dataset"]), 
                      len(training_type2split2source2dataset_or_loader[training_type]["val"][source]["dataset"]),
                      len(training_type2split2source2dataset_or_loader[training_type]["test"][source]["dataset"])])
            print("raw: {}".format(dataset_split_size_list))
            print("frac: {}".format(dataset_split_size_list/np.sum(dataset_split_size_list)))
            print()

# def train_loop(training_type, sourceint_list, model, args, device, training_type2split2source2dataset_or_loader, source2task, results_dict, split2writer):
#     # Originally written to enable pretraining on one tabular dataset and fine-tuning on a different tabular dataset. In the end, we just use it to finetune on a single tabular dataset
#     # The pretraining mentioned in the PLATO manuscript refers to generation of the node embeddings from the KG; that is NOT the same pretraining described in the code below.
#     assert(training_type == "finetune")

#     # Set up optimizer
#     print("{}...".format(training_type))
#     print(f"lr: {args.lr}")
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # Get initial performance
#     assert(len(sourceint_list) == 1) # BUG: Will break pretraining (okay for now, just finetuning); need to review all sourceint mentions to restore pretraining
#     sourceint = sourceint_list[0]
#     # Get initial performance
#     _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["train"], epoch = 0)
#     _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["test"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["test"], epoch = 0)
#     _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["val"], epoch = 0)

#     # Main training loop
#     best_avg_val_score_across_sources = -1*float('inf')
#     pbar = tqdm(range(1, 1 + args.epochs))
#     train_pr, val_pr = 0, 0
#     for epoch in pbar:
#         # Train model (one source only)
#         assert(len(sourceint_list) == 1)
#         pbar.set_description("epoch: %d, train PR: %.4f, val PR: %.4f" % (epoch, train_pr, val_pr))
#         train(model, device, training_type2split2source2dataset_or_loader[training_type]["train"][sourceint_list[0]]["loader"], optimizer)

#         # Validate (originally written to account for multiple sources but in the end, our problem setting is such that there's only ever a single source). Save performance
#         if epoch % args.log_steps == 0:
#             # Measure the performance of the model on the validation set
#             # 1. Measure validation performance on each source separately
#             val_source2metric2score = dict()
#             for sourceint in sourceint_list:
#                 # Get performance for each on training and test (for tensorboard writer only)
#                 train_metric2score, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["train"], epoch = epoch)
#                 # print("train loss, R2: {:.4f}, {:.4f}".format(-1*train_metric2score["neg_loss"], train_metric2score["pearsonr"]**2))
#                 train_pr = train_metric2score["pearsonr"]
#                 _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["test"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["test"], epoch = epoch)

#                 # Get performance for each source on validation
#                 metric2score, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["val"], epoch = epoch)
#                 val_pr = metric2score["pearsonr"]
#                 val_source2metric2score[sourceint] = metric2score

#             # 2. Select best model by taking average of validation score across the sources
#             avg_val_loss_across_sources = np.mean([-1*metric2score["neg_loss"] for source, metric2score in val_source2metric2score.items()])
#             avg_val_score_across_sources = np.mean([metric2score[args.selection_metric] for source, metric2score in val_source2metric2score.items()])
#             # print('This Run\'s Average Val Loss, Score Across Sources: {:.2f}, {:.2f}'.format(avg_val_loss_across_sources, avg_val_score_across_sources))

#             # If average validation score of the model across sources is the best seen so far, then...
#             if avg_val_score_across_sources > best_avg_val_score_across_sources:
#                 # Save the model and best average validation score
#                 best_avg_val_score_across_sources = avg_val_score_across_sources
#                 best_model_state_dict = copy.deepcopy(model.state_dict())

#                 # print("New best_avg_val_score_across_sources: {}".format(best_avg_val_score_across_sources))
                
#     # Load the best model
#     model.load_state_dict(best_model_state_dict)

#     # Measure performance of the best model on all sources and splits
#     source2split2metric2score = dict()
#     source2split2vec_name2vec = dict()
#     for sourceint in sourceint_list:
#         split2metric2score = dict()
#         split2vec_name2vec = dict()
#         for split in SPLITS:
#             vec_name2vec = dict()
#             metric2score, vec_name2vec["y_prob"], vec_name2vec["y_true"], vec_name2vec["drugs"], vec_name2vec["entities"], vec_name2vec["sources"] = eval(model, device, training_type2split2source2dataset_or_loader[training_type][split][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer[split], epoch = epoch)
#             split2metric2score[split] = metric2score
#             split2vec_name2vec[split] = vec_name2vec
#         source2split2metric2score[sourceint] = split2metric2score
#         source2split2vec_name2vec[sourceint] = split2vec_name2vec

#     # Print performance of best model
#     print("Overall performance...")
#     print(f'best avg val {args.selection_metric} across sources', best_avg_val_score_across_sources)

#     # Update results_dict
#     assert(not(training_type in results_dict))
#     results_dict[training_type] = {"source2split2metric2score": source2split2metric2score, "source2split2vec_name2vec": source2split2vec_name2vec}

#     return model, results_dict


# def train_loop(training_type, sourceint_list, model, args, device, training_type2split2source2dataset_or_loader, source2task, results_dict, split2writer):
#     # Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    
#     assert len(sourceint_list) == 1
#     sourceint = sourceint_list[0]
    
#     train_loader = training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["loader"]
    
#     best_val_score = -np.inf
#     best_epoch = -1
    
#     # === 修改 1: 恢复用户要求的 range(1, 1 + args.epochs) ===
#     pbar = tqdm(range(1, 1 + args.epochs))
    
#     for epoch in pbar:
#         model.train()
        
#         # Training Step
#         for x, y, loader_sourceint_list, entity_idx, drug_idxs in train_loader:
#             x, y, source = x.to(device), y.to(device), loader_sourceint_list.to(device)
            
#             optimizer.zero_grad()
#             loss = model.compute_loss(x, y, source)
            
#             # L1 Regularization
#             if args.l1_weight > 0:
#                 l1_reg = torch.tensor(0., device=device)
#                 for param in model.parameters():
#                     l1_reg += torch.norm(param, 1)
#                 loss += args.l1_weight * l1_reg
                
#             loss.backward()
#             optimizer.step()
            
#         # Evaluation Step
#         # === 修改 2: 由于 epoch 已经是从 1 开始的，所以不需要 +1 ===
#         if epoch % args.log_steps == 0:
#             train_metric2score, _, _, _, _, _ = eval(model, device, train_loader, sourceint, source2task, split2writer["train"], epoch)
            
#             val_loader = training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"]
#             val_metric2score, _, _, _, _, _ = eval(model, device, val_loader, sourceint, source2task, split2writer["val"], epoch)
            
#             # === 修改 3: 动态选择进度条显示的指标 (解决 KeyError: pearsonr) ===
#             if "pearsonr" in train_metric2score:
#                 metric_key = "pearsonr"
#                 disp_name = "PR"
#             elif "auroc" in train_metric2score:
#                 metric_key = "auroc"
#                 disp_name = "AUC"
#             elif "acc" in train_metric2score:
#                 metric_key = "acc"
#                 disp_name = "ACC"
#             else:
#                 # 兜底：如果没有找到标准指标，打印 Loss 或第一个 key
#                 metric_key = list(train_metric2score.keys())[0] if train_metric2score else "loss"
#                 disp_name = metric_key

#             train_disp = train_metric2score.get(metric_key, 0.0)
#             val_disp = val_metric2score.get(metric_key, 0.0)
            
#             pbar.set_description(f"epoch: {epoch}, train {disp_name}: {train_disp:.4f}, val {disp_name}: {val_disp:.4f}")
            
#             # === 修改 4: 自动纠正 selection_metric (防止 neg_mse 报错) ===
#             if args.selection_metric not in val_metric2score:
#                 old_metric = args.selection_metric
#                 args.selection_metric = metric_key # 切换为当前存在的指标
#                 # 仅在第一轮打印警告
#                 if epoch == 1: # 因为从 1 开始
#                     print(f"\n[Warning] Selection metric '{old_metric}' not found. Switching to '{args.selection_metric}'.")

#             # Checkpointing
#             if val_metric2score[args.selection_metric] > best_val_score:
#                 best_val_score = val_metric2score[args.selection_metric]
#                 best_epoch = epoch
                
#                 # Save best model logic (simplified)
#                 # torch.save(model.state_dict(), args.filename) 
                
#     # End of training
#     # Save results to dict
    
#     # Store final results
#     # results_dict[training_type]['agg_source2split2metric2score'][sourceint]['val'] = val_metric2score
    
#     # # Evaluate on test set
#     # test_loader = training_type2split2source2dataset_or_loader[training_type]["test"][sourceint]["loader"]
#     # # 注意：这里 epoch 传 args.epochs，或者传 best_epoch
#     # test_metric2score, _, _, _, _, _ = eval(model, device, test_loader, sourceint, source2task, split2writer["test"], args.epochs)
#     # results_dict[training_type]['agg_source2split2metric2score'][sourceint]['test'] = test_metric2score

#     # === 修改开始：正确初始化字典并保存结果 ===
    
#     # 1. 初始化 training_type 键 (例如 'finetune')
#     if training_type not in results_dict:
#         results_dict[training_type] = {
#             "source2split2metric2score": {},
#             "source2split2vec_name2vec": {} # 预留结构，防止 save_results 报错
#         }

#     # 2. 初始化 sourceint 键
#     if sourceint not in results_dict[training_type]["source2split2metric2score"]:
#         results_dict[training_type]["source2split2metric2score"][sourceint] = {}

#     # 3. 保存验证集结果 (Val)
#     results_dict[training_type]["source2split2metric2score"][sourceint]['val'] = val_metric2score
    
#     # 4. 计算并保存测试集结果 (Test)
#     test_loader = training_type2split2source2dataset_or_loader[training_type]["test"][sourceint]["loader"]
#     # 注意：这里 epoch 传 args.epochs 即可
#     test_metric2score, _, _, _, _, _ = eval(model, device, test_loader, sourceint, source2task, split2writer["test"], args.epochs)
    
#     results_dict[training_type]["source2split2metric2score"][sourceint]['test'] = test_metric2score

#     # 5. (可选) 保存训练集结果，防止某些绘图代码报错
#     # 如果您不需要训练集最终指标，这步可以省略，但在 PLATO 中最好加上
#     results_dict[training_type]["source2split2metric2score"][sourceint]['train'] = train_metric2score

#     # === 修改结束 ===

#     return model, results_dict

def train_loop(training_type, sourceint_list, model, args, device, training_type2split2source2dataset_or_loader, source2task, results_dict, split2writer):
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    
    assert len(sourceint_list) == 1
    sourceint = sourceint_list[0]
    
    train_dataset = training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["dataset"]
    train_loader = training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["loader"]
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # === [核心修改] 自动计算并应用类别平衡权重 ===
    current_task = get_task(sourceint, source2task)
    if current_task == "multilabel":
        # === 根据命令行参数决定是否使用类别权重 ===
        if getattr(args, 'use_class_weights', False):
            # 1. 计算权重
            pos_weight = compute_class_weights(train_eval_loader, device)
            
            # 2. 更新模型的 Loss 函数
            # 注意：model.loss_list 是一个列表，索引对应 sourceint
            print(f"Updating BCE loss for source {sourceint} with pos_weight.")
            # 创建带权重的 Loss
            weighted_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            # 替换原有 Loss
            model.loss_list[sourceint] = weighted_loss_fn
        else:
            print(f"\n>>> [INFO] ⚪ 关闭类别权重: Using standard BCE loss for source {sourceint} <<<")
    # ==========================================

    adaptive_negative_sampling = (
        current_task == "multilabel"
        and getattr(args, 'use_adaptive_negative_sampling', False)
    )
    disease_typed_bidirectional_sampling = (
        current_task == "multilabel"
        and getattr(args, 'use_disease_typed_bidirectional_sampling', False)
    )
    adaptive_bidirectional_sampling = (
        current_task == "multilabel"
        and (
            getattr(args, 'use_bidirectional_adaptive_sampling', False)
            or disease_typed_bidirectional_sampling
        )
    )
    if adaptive_negative_sampling and adaptive_bidirectional_sampling:
        print("\n>>> [INFO] 同时开启了自适应阴性采样和双向自适应采样，本次优先使用双向自适应采样。<<<")
        adaptive_negative_sampling = False
    adaptive_neg_alpha = None
    adaptive_bidir_alpha_pos = None
    adaptive_bidir_alpha_neg = None
    adaptive_bidir_prevalence = None
    if adaptive_negative_sampling:
        # 为每个疾病标签初始化一个受反馈控制的采样强度；实际难阴性集合会在每次验证后重新估计。
        y_train_init, _ = collect_multilabel_predictions(
            model, device, train_eval_loader, sourceint, source2task
        )
        adaptive_neg_alpha = np.ones(y_train_init.shape[1], dtype=float)
        print(
            "\n>>> [INFO] 已启用自适应难阴性采样，"
            f"覆盖 {y_train_init.shape[1]} 个疾病标签。 <<<"
        )

    if adaptive_bidirectional_sampling:
        # 双向机制为每个疾病分别维护阳性采样强度和阴性采样强度。
        y_train_init, _ = collect_multilabel_predictions(
            model, device, train_eval_loader, sourceint, source2task
        )
        adaptive_bidir_alpha_pos = np.ones(y_train_init.shape[1], dtype=float)
        adaptive_bidir_alpha_neg = np.ones(y_train_init.shape[1], dtype=float)
        adaptive_bidir_prevalence = y_train_init.mean(axis=0)
        sampler_name = "疾病分型驱动的双向采样" if disease_typed_bidirectional_sampling else "双向自适应采样"
        print(
            f"\n>>> [INFO] 已启用{sampler_name}，"
            f"覆盖 {y_train_init.shape[1]} 个疾病标签，"
            f"阳性占比范围 {adaptive_bidir_prevalence.min():.3f}-{adaptive_bidir_prevalence.max():.3f}。<<<"
        )

    best_val_score = -np.inf
    best_epoch = -1
    best_model_state_dict = None # 用于存储最佳模型参数
    
    pbar = tqdm(range(1, 1 + args.epochs))
    
    for epoch in pbar:
        model.train()
        
        # Training Step
        for x, y, loader_sourceint_list, entity_idx, drug_idxs in train_loader:
            x, y, source = x.to(device), y.to(device), loader_sourceint_list.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(x, y, source)
            
            # L1 Regularization
            if args.l1_weight > 0:
                l1_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += args.l1_weight * l1_reg
                
            loss.backward()
            optimizer.step()
            
        # Evaluation Step
        if epoch % args.log_steps == 0:
            train_metric2score, _, _, _, _, _, _ = eval(model, device, train_eval_loader, sourceint, source2task, split2writer["train"], epoch)
            
            val_loader = training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"]
            val_metric2score, val_prob, val_true, _, _, _, _ = eval(model, device, val_loader, sourceint, source2task, split2writer["val"], epoch)
            
            # 动态选择显示的指标
            if "pearsonr" in train_metric2score:
                metric_key = "pearsonr"
                disp_name = "PR"
            elif "balanced_macro_score" in train_metric2score:
                metric_key = "balanced_macro_score"
                disp_name = "Balanced-Macro"
            elif "auroc" in train_metric2score:
                metric_key = "auroc"
                disp_name = "AUC"
            elif "acc" in train_metric2score:
                metric_key = "acc"
                disp_name = "ACC"
            else:
                metric_key = list(train_metric2score.keys())[0] if train_metric2score else "loss"
                disp_name = metric_key

            train_disp = train_metric2score.get(metric_key, 0.0)
            val_disp = val_metric2score.get(metric_key, 0.0)
            
            pbar.set_description(f"epoch: {epoch}, train {disp_name}: {train_disp:.4f}, val {disp_name}: {val_disp:.4f}")
            
            # 自动纠正 selection_metric
            if args.selection_metric not in val_metric2score:
                old_metric = args.selection_metric
                args.selection_metric = metric_key 
                if epoch == 1:
                    print(f"\n[Warning] Selection metric '{old_metric}' not found. Switching to '{args.selection_metric}'.")

            # Checkpointing: 保存最佳模型状态
            if val_metric2score[args.selection_metric] > best_val_score:
                best_val_score = val_metric2score[args.selection_metric]
                best_epoch = epoch
                # 使用 deepcopy 保存当前最佳参数
                best_model_state_dict = copy.deepcopy(model.state_dict())

            if (
                adaptive_bidirectional_sampling
                and epoch >= args.adaptive_bidir_warmup_epochs
                and epoch % args.adaptive_bidir_update_freq == 0
            ):
                # 验证集决定每个疾病下一轮更应该增强阳性还是增强阴性；
                # 训练集反馈只用于识别 train-val 落差，避免重复采样造成过拟合。
                feedback = compute_multilabel_feedback(val_true.numpy(), val_prob.numpy())
                y_train_true, y_train_prob = collect_multilabel_predictions(
                    model, device, train_eval_loader, sourceint, source2task
                )
                train_feedback = compute_multilabel_feedback(y_train_true, y_train_prob)
                if disease_typed_bidirectional_sampling:
                    adaptive_bidir_alpha_pos, adaptive_bidir_alpha_neg, _, disease_typed_params = update_disease_typed_bidirectional_sampling_alpha(
                        adaptive_bidir_alpha_pos,
                        adaptive_bidir_alpha_neg,
                        feedback,
                        args,
                        adaptive_bidir_prevalence,
                        train_feedback,
                    )
                    train_loader = build_disease_typed_bidirectional_loader(
                        train_dataset,
                        args.batch_size,
                        y_train_true,
                        y_train_prob,
                        adaptive_bidir_alpha_pos,
                        adaptive_bidir_alpha_neg,
                        disease_typed_params,
                        args,
                    )
                else:
                    adaptive_bidir_alpha_pos, adaptive_bidir_alpha_neg = update_bidirectional_sampling_alpha(
                        adaptive_bidir_alpha_pos,
                        adaptive_bidir_alpha_neg,
                        feedback,
                        args,
                        adaptive_bidir_prevalence,
                        train_feedback,
                    )
                    # 训练集重新打分后，只对当前模型容易错的难阳性/难阴性样本提高采样概率。
                    train_loader = build_bidirectional_adaptive_loader(
                        train_dataset,
                        args.batch_size,
                        y_train_true,
                        y_train_prob,
                        adaptive_bidir_alpha_pos,
                        adaptive_bidir_alpha_neg,
                        args,
                    )

            if (
                adaptive_negative_sampling
                and epoch >= args.adaptive_neg_warmup_epochs
                and epoch % args.adaptive_neg_update_freq == 0
            ):
                # 根据验证集反馈决定下一轮训练中每个疾病标签应多大程度强调难阴性样本。
                feedback = compute_multilabel_feedback(val_true.numpy(), val_prob.numpy())
                adaptive_neg_alpha = update_negative_sampling_alpha(
                    adaptive_neg_alpha, feedback, args
                )
                # 用不打乱顺序的训练集视图重新打分，确保采样权重与 dataset 顺序对齐。
                y_train_true, y_train_prob = collect_multilabel_predictions(
                    model, device, train_eval_loader, sourceint, source2task
                )
                train_loader = build_adaptive_negative_loader(
                    train_dataset,
                    args.batch_size,
                    y_train_true,
                    y_train_prob,
                    adaptive_neg_alpha,
                    args,
                )

    # === 训练结束 ===
    
    # 1. 加载最佳模型 (如果存在)
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        # print(f"Loaded best model from epoch {best_epoch} (Score: {best_val_score:.4f})")

    decision_thresholds = None
    threshold_info = None
    if current_task == "multilabel" and getattr(args, "use_disease_threshold_optimization", False):
        # 只用验证集搜索每个疾病的判阳阈值，测试集只用于最终评估，避免信息泄露。
        val_loader = training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"]
        _, val_prob_for_threshold, val_true_for_threshold, _, _, _, _ = eval(
            model,
            device,
            val_loader,
            sourceint,
            source2task,
            split2writer["val"],
            args.epochs,
        )
        decision_thresholds, threshold_info = optimize_typed_multilabel_decision_thresholds(
            val_true_for_threshold.numpy(),
            val_prob_for_threshold.numpy(),
            args,
        )

    # 2. 初始化结果字典结构
    if training_type not in results_dict:
        results_dict[training_type] = {
            "source2split2metric2score": {},
            "source2split2vec_name2vec": {}
        }

    if sourceint not in results_dict[training_type]["source2split2metric2score"]:
        results_dict[training_type]["source2split2metric2score"][sourceint] = {}
        results_dict[training_type]["source2split2vec_name2vec"][sourceint] = {}

    # 3. 重新评估所有数据集 (Train/Val/Test) 并保存详细向量
    #    这一步会填充 source2split2vec_name2vec
    for split in ['train', 'val', 'test']:
        loader = training_type2split2source2dataset_or_loader[training_type][split][sourceint]["loader"]
        
        # 运行 eval 获取所有返回值
        metric2score, y_prob, y_true, drug_list, entity_list, source_list, patient_scores = eval(
            model,
            device,
            loader,
            sourceint,
            source2task,
            split2writer[split],
            args.epochs,
            decision_thresholds=decision_thresholds,
        )
        
        # 保存指标
        results_dict[training_type]["source2split2metric2score"][sourceint][split] = metric2score
        
        # # 保存向量 (这就是您之前缺失的部分)
        # results_dict[training_type]["source2split2vec_name2vec"][sourceint][split] = {
        #     "y_prob": y_prob,     # 预测概率
        #     "y_true": y_true,     # 真实标签
        #     "drugs": drug_list,   # 药物ID
        #     "entities": entity_list, # 实体(样本)ID
        #     "sources": source_list # 数据来源
        # }
        # === 将 patient_scores 合并到保存的向量字典中 ===
        # 这样在 results/save.pt 中就能找到这些患者的综合评分
        vec_dict = {
            "y_prob": y_prob,
            "y_true": y_true,
            "drugs": drug_list,
            "entities": entity_list,
            "sources": source_list
        }
        vec_dict.update(patient_scores) # 合并 CHD_Diagnosis_prob, Coronary_Abnormal_prob
        
        if decision_thresholds is not None:
            vec_dict["decision_thresholds"] = torch.tensor(decision_thresholds, dtype=torch.float32)
            vec_dict["threshold_info"] = threshold_info
        results_dict[training_type]["source2split2vec_name2vec"][sourceint][split] = vec_dict

    return model, results_dict

def add_to_agg_source2split2metric2score(agg_source2split2metric2score, source, split, metric, score):
    if source in agg_source2split2metric2score:
        if split in agg_source2split2metric2score[source]:
            if metric in agg_source2split2metric2score[source][split]:
                agg_source2split2metric2score[source][split][metric].append(score)
            else:
                agg_source2split2metric2score[source][split][metric] = [score]
        else:
            agg_source2split2metric2score[source][split] = {metric : [score]}
    else:
        agg_source2split2metric2score[source] = {split: {metric: [score]}}
    return agg_source2split2metric2score

def agg_source2split2metric2score(results_dict_list, training_type):
    agg_source2split2metric2score = dict()
    for results_dict in results_dict_list:
        source2split2metric2score = results_dict[training_type]["source2split2metric2score"]
        for source, split2metric2score in source2split2metric2score.items():
            for split, metric2score in split2metric2score.items():
                for metric, score in metric2score.items():
                    agg_source2split2metric2score = add_to_agg_source2split2metric2score(agg_source2split2metric2score, source, split, metric, score)
    
    return agg_source2split2metric2score

def summarize_agg_metric_lists(agg_dict):
    """
    agg_dict: source -> split -> metric -> [score_run0, score_run1, ...]
    return: (mean_dict, std_dict) with same nesting but metric -> float
    """
    mean_dict = {}
    std_dict = {}
    for source, split2metric in agg_dict.items():
        mean_dict[source] = {}
        std_dict[source] = {}
        for split, metric2list in split2metric.items():
            mean_dict[source][split] = {}
            std_dict[source][split] = {}
            for metric, scores in metric2list.items():
                # scores 可能是 tensor/float 混合，先转成 float list
                vals = []
                for s in scores:
                    if hasattr(s, "detach"):
                        s = s.detach()
                    if hasattr(s, "item"):
                        try:
                            s = s.item()
                        except Exception:
                            pass
                    vals.append(float(s))
                vals = np.array(vals, dtype=float)
                valid_vals = vals[~np.isnan(vals)]
                mean_dict[source][split][metric] = float(np.mean(valid_vals)) if len(valid_vals) else float("nan")
                std_dict[source][split][metric] = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0
    return mean_dict, std_dict

def agg_split2y_pred(results_dict_list, training_type):
    split2y_pred_list = []
    for results_dict in results_dict_list:
        split2y_pred_list.append(results_dict[training_type]["split2y_pred"])
    return split2y_pred_list

def agg_split2const_list(results_dict_list, training_type, k):
    # Keys with const lists
    assert(k in ["split2y_true", "split2sources", "split2drugs", "split2entities"])

    # Ensure that all of them are truly the same
    const_list = None
    for results_dict in results_dict_list:
        assert(not(results_dict[training_type][k] is None))
        if const_list is None:
            const_list = results_dict[training_type][k]
        assert(const_list == results_dict[training_type][k])

    return const_list

def validate_keys_in_list_of_eq_dicts(dict_list):
    # Ensure that all of the results_dicts have the same keys (must flatten first to compare structure)
    comp_keys = None
    for dict_ in dict_list:
        keys_i = set(flatten_nested_dict(dict_).keys())
        if comp_keys is None:
            comp_keys = keys_i
        assert(comp_keys == keys_i)

def save_results(results_dict_list, training_type2split2source2dataset_or_loader, args, splits):
    # Ensure that all of the results_dicts have the same keys (must flatten first to compare structure)
    validate_keys_in_list_of_eq_dicts(results_dict_list)

    # For each key, iterate over the results dicts, aggregate, and populate agg_results_dict
    rep_results_dict = results_dict_list[0]
    agg_results_dict = {k: dict() for k in rep_results_dict.keys()}
    for training_type in agg_results_dict.keys():
        for k in rep_results_dict[training_type].keys():
            # if k == "source2split2metric2score":
            #     agg_results_dict[training_type]["agg_" + k] = agg_source2split2metric2score(results_dict_list, training_type)
            if k == "source2split2metric2score":
                agg_list = agg_source2split2metric2score(results_dict_list, training_type)
                agg_results_dict[training_type]["agg_" + k] = agg_list

                mean_dict, std_dict = summarize_agg_metric_lists(agg_list)
                agg_results_dict[training_type]["agg_" + k + "_mean"] = mean_dict
                agg_results_dict[training_type]["agg_" + k + "_std"] = std_dict
            elif k == "split2y_pred":
                agg_results_dict[training_type][k + "_list"] = agg_split2y_pred(results_dict_list, training_type)
            elif k in ["split2y_true", "split2sources", "split2drugs", "split2entities"]:
                agg_results_dict[training_type][k] = agg_split2const_list(results_dict_list, training_type, k)
            # === 关键修改：保存包含患者级评分的向量字典 ===
            elif k == "source2split2vec_name2vec":
                # 保存第一个 Run 的向量结果，通常这就够了
                agg_results_dict[training_type][k] = rep_results_dict[training_type][k]
            # ============================================
            else:
                # TODO: Add functionality (used to print "MISSING FUNCTIONALITY")
                print(k)

    # Generate entity and drug mappings and add
    idx2entity_map = dict()
    idx2drug_map = dict()
    for split in splits:
        assert(len(args.finetune_sourceint_list) == 1)
        dataset = training_type2split2source2dataset_or_loader["finetune"][split][args.finetune_sourceint_list[0]]["dataset"]
        idx2entity_map[split] = dataset.entity_ids
        idx2drug_map[split] = dataset.drug_names
    agg_results_dict['idx2entity'] = idx2entity_map
    agg_results_dict['idx2drug'] = idx2drug_map

    # Add input arguments
    agg_results_dict["args"] = vars(args)

    # Save
    if args.filename != '':
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.filename), exist_ok=True)
        torch.save(agg_results_dict, args.filename)

    return agg_results_dict

def get_gene_ixs(x, gene_dim):
    return x[:, :gene_dim]

def get_drug_ixs(x, gene_dim):
    return x[:, gene_dim:]

def get_gene_embed(provider):
    # Matrix where every row corresponds to a gene embedding. Rows are ordered in the same way as the gene expression input vector
    gene_embed = provider.kg.data.x[provider.kg_mapping_dict['X_expression']] # (n_genes, d)
    # dt(gene_embed, "gene_embed")

    return gene_embed

def get_drug_embed(provider):
    # Matrix where every row corresponds to a drug embedding. Rows are ordered in the same way as in the multihot drug input vector
    drug_embed = provider.kg.data.x[provider.kg_mapping_dict['drug']]

    return drug_embed

class ParentDecoder(torch.nn.Module):
    def __init__(self):
        super(ParentDecoder, self).__init__()

    # def get_loss_list(self):
    #     # Set loss according to regression task
    #     reg_criterion = torch.nn.MSELoss(reduction = 'mean')

    #     # Populate loss list
    #     loss_list = []
    #     for source in ["cell", "mouse", "patient"]:
    #         assert(self.source2task[source] == "numeric")
    #         loss_list.append(reg_criterion)

    #     return loss_list
    def get_loss_list(self):
        # Populate loss list
        loss_list = []
        # source2task 可能包含整数键(如0)或字符串键("cell")，根据您的 dataset 设置
        # 这里为了兼容，我们在循环中判断
        
        # 注意：这里原代码是硬编码 ["cell", "mouse", "patient"]
        # 我们假设自定义数据的 sourceint 都在 self.source2task 中
        
        # 为了兼容原代码逻辑，我们构建一个按 sourceint 顺序的 loss 列表
        # 假设最大 sourceint 不超过 10
        max_source = 10 
        for source_id in range(max_source):
            # 尝试获取任务类型，先找 sourceint，再找对应的名字
            task = None
            if source_id in self.source2task:
                task = self.source2task[source_id]
            elif source_id in INT2SOURCE and INT2SOURCE[source_id] in self.source2task:
                task = self.source2task[INT2SOURCE[source_id]]
            
            if task == "numeric":
                loss_list.append(torch.nn.MSELoss(reduction = 'mean'))
            elif task == "multilabel":
                loss_list.append(torch.nn.BCEWithLogitsLoss(reduction = 'mean'))
            else:
                # 默认填充，防止索引越界，或者你可以 append None
                loss_list.append(torch.nn.MSELoss(reduction = 'mean'))

        return loss_list

    # def compute_loss_without_reg(self, x, y, sourceint_list):
    #     '''
    #         Calculates loss across samples.

    #         sourceint_list (torch tensor): stores where i-th element comes from 
    #             - 0: cell line
    #             - 1: mouse
    #             - 2: patient
    #     '''
    #     loss_total = 0
    #     unique_sourceint_list = torch.unique(sourceint_list)

    #     # Handle one source and therefore one task (i.e. regression) at a time
    #     for sourceint in unique_sourceint_list:
    #         # true-false vector
    #         tf_idx = sourceint_list == sourceint

    #         # extracting features for sourceint
    #         x_sub, y_sub = x[tf_idx], y[tf_idx]

    #         pred = self(x_sub, sourceint)

    #         assert(get_task(sourceint.item(), self.source2task) == "numeric") # Need the .item() here because coming from Torch tensor
    #         pred = pred.view(-1,) # Convert from (n, 1) to (n) so loss computed correctly

    #         assert(pred.shape == y_sub.shape)
    #         loss = self.loss_list[int(sourceint.item())](pred, y_sub)
            
    #         loss_total += loss
        
    #     return loss_total
    def compute_loss_without_reg(self, x, y, sourceint_list):
        loss_total = 0
        unique_sourceint_list = torch.unique(sourceint_list)

        for sourceint in unique_sourceint_list:
            tf_idx = sourceint_list == sourceint
            x_sub, y_sub = x[tf_idx], y[tf_idx]

            pred = self(x_sub, sourceint)
            
            # === 修改逻辑: 获取任务类型 ===
            task_type = "numeric" # 默认
            s_item = int(sourceint.item())
            if s_item in self.source2task:
                task_type = self.source2task[s_item]
            elif s_item in INT2SOURCE and INT2SOURCE[s_item] in self.source2task:
                task_type = self.source2task[INT2SOURCE[s_item]]
            # ============================

            # ====== 新增：在计算 Loss 前，进行规则拦截 ======
            # (这会让模型由于梯度回传，在权重更新上更偏向于遵守医学逻辑)
            if getattr(self, 'use_rule_mask', False):
                pred = apply_hard_rules(x_sub, pred, task_type, getattr(self, 'feat_indices', {}))
            # ==============================================

            if task_type == "numeric":
                pred = pred.view(-1,) 
            
            # 如果是 multilabel，pred 形状是 (N, Classes)，y_sub 也是 (N, Classes)，不需要 view(-1)

            # 确保使用对应的 Loss 函数
            # loss_list 是列表，通过 int(sourceint) 索引
            loss = self.loss_list[s_item](pred, y_sub)
            
            loss_total += loss
        
        return loss_total

    def compute_loss(self, x, y, sourceint_list):
        # Get sum without regularization
        loss = self.compute_loss_without_reg(x, y, sourceint_list)
        
        # Get l1, l2 regularizations
        l1_loss = 0.0
        l2_loss = 0.0
        for name, parameter in self.named_parameters():
            l1_loss += get_l1_reg(parameter, self.l1_weight)
            l2_loss += get_l2_reg(parameter, self.l2_weight)

        # Sum
        loss = loss + l1_loss + l2_loss

        return loss
