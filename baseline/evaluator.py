from collections import OrderedDict
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, jaccard_score, hamming_loss, precision_score, recall_score, f1_score

class Eval():
	def __init__(self):
		pass

	def assert_inputs_are_numpy_arrays(self, input_list):
		# Assert that inputs are numpy arrays
		for array in input_list:
			assert(type(array) is np.ndarray)

	def evaluate_all(self):
		raise NotImplementedError

class RegEval(Eval):
	'''
	This class evaluates the predicted values of a regression model
	'''
	def __init__(self):
		pass

	def neg_mse(self, y_true, y_pred):
		'''
		Returns the negative of the mean squared error between y_pred and y_true.
		'''
		neg_mse = -1*float((np.sum(np.square(y_pred - y_true)) / len(y_pred)).item())
		return neg_mse

	def pearsonr(self, y_true, y_pred):
		'''
		Returns the Pearson correlation between y_true and y_pred.
		'''
		r, p = pearsonr(y_true, y_pred)
		return r, p

	def evaluate_all(self, y_true, y_pred):
		# Assert inputs are numpy arrays
		self.assert_inputs_are_numpy_arrays([y_true, y_pred])

		# Create metric2score dictionary with regression metrics
		metric2score = OrderedDict()
		metric2score["neg_mse"] = self.neg_mse(y_true, y_pred)
		metric2score["pearsonr"], metric2score["pearsonp"] = self.pearsonr(y_true, y_pred)

		return metric2score, dict()

# === 多标签 + 风险分级评估器 ===
class MultilabelEval(Eval):
    '''
    This class evaluates the predicted values of a multilabel classification model
    '''
    def __init__(self, t_low=0.3, t_high=0.7, decision_thresholds=None):
        # pass
        self.t_low = t_low
        self.t_high = t_high
        # 每个疾病可以使用不同的判阳阈值；不传入时保持原来的 0.5 统一阈值。
        self.decision_thresholds = decision_thresholds

    def evaluate_all(self, y_true, y_pred):
        # y_true: (N, num_classes) 0/1 labels
        # y_pred: (N, num_classes) Logits (raw output from model)
        
        self.assert_inputs_are_numpy_arrays([y_true, y_pred])

		# === 修复：确保 y_true 是整数类型，防止 sklearn 报错 ===
        y_true = y_true.astype(int) 
        # ======================================================
        
        metric2score = OrderedDict()

        # 用于存储患者级的具体评分 (N,) 数组
        patient_scores = {}
        
        # 1. 将 Logits 转换为概率 (Sigmoid)
        y_prob = 1 / (1 + np.exp(-y_pred))
        
        # 2. 将概率转换为二值预测 (Threshold = 0.5)
        if self.decision_thresholds is None:
            thresholds = np.full(y_prob.shape[1], 0.5, dtype=float)
        else:
            thresholds = np.asarray(self.decision_thresholds, dtype=float)
            if thresholds.ndim == 0:
                thresholds = np.full(y_prob.shape[1], float(thresholds), dtype=float)
            if thresholds.shape[0] != y_prob.shape[1]:
                raise ValueError(
                    f"decision_thresholds length {thresholds.shape[0]} does not match num_classes {y_prob.shape[1]}"
                )
        # 默认阈值为 0.5；开启疾病级阈值优化后，每个疾病使用各自的阈值。
        y_pred_binary = (y_prob >= thresholds.reshape(1, -1)).astype(int)

        # 3. 计算 AUROC (Micro average 适用于多标签)
        try:
            metric2score["auroc_micro"] = roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            # 防止因某一类全为0导致报错
            metric2score["auroc_micro"] = float("nan")
        try:
            metric2score["auroc_samples"] = roc_auc_score(y_true, y_prob, average='samples')
        except ValueError:
            metric2score["auroc_samples"] = float("nan")
        # AUPRC/AP is computed from continuous probabilities, never thresholded labels.
        try:
            metric2score["auprc_micro"] = average_precision_score(
                y_true, y_prob, average='micro'
            )
        except ValueError:
            metric2score["auprc_micro"] = float("nan")
        # metric2score["auroc"] = metric2score["auroc_micro"]
            
        # 4. 计算 Accuracy
        # 在多标签中，accuracy_score 指的是 subset accuracy (所有标签都预测对才算对)
        # 如果您想要 hamming score，可以另行实现
        # metric2score["acc"] = accuracy_score(y_true, y_pred_binary)
        # metric2score["acc"] = (y_true == y_pred_binary).mean()
        h_loss = hamming_loss(y_true, y_pred_binary)
        metric2score["acc"] = 1 - h_loss

        # 使用 average='samples' 计算每个病人的平均表现，zero_division=0 防止报错
        # Micro metrics summarize all patient-disease label decisions together.
        metric2score["precision_micro"] = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metric2score["recall_micro"] = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metric2score["f1_micro"] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metric2score["jaccard_micro"] = jaccard_score(y_true, y_pred_binary, average='micro', zero_division=0)

        # Sample metrics summarize each patient's label set first, then average across patients.
        metric2score["precision_samples"] = precision_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metric2score["recall_samples"] = recall_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metric2score["f1_samples"] = f1_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metric2score["jaccard_samples"] = jaccard_score(y_true, y_pred_binary, average='samples', zero_division=1)

        # Backward-compatible aliases. These are sample-averaged.
        # metric2score["precision"] = metric2score["precision_samples"]
        # metric2score["recall"] = metric2score["recall_samples"]
        # metric2score["f1"] = metric2score["f1_samples"]

        jaccard_per = jaccard_score(y_true, y_pred_binary, average=None, zero_division=1)
        precision_per = precision_score(y_true, y_pred_binary, average=None, zero_division=0)
        recall_per = recall_score(y_true, y_pred_binary, average=None, zero_division=0)
        f1_per = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
        # 手动计算每列的 Accuracy: (TP + TN) / Total
        acc_per = (y_true == y_pred_binary).mean(axis=0)

        # 基于混淆矩阵计算每个疾病的阴性识别能力。
        # 某个 split 中如果没有阴性或没有阳性，对应指标记为 nan，避免用 0 掩盖数据不可定义问题。
        tp_per = np.sum((y_true == 1) & (y_pred_binary == 1), axis=0)
        tn_per = np.sum((y_true == 0) & (y_pred_binary == 0), axis=0)
        fp_per = np.sum((y_true == 0) & (y_pred_binary == 1), axis=0)
        fn_per = np.sum((y_true == 1) & (y_pred_binary == 0), axis=0)
        neg_denom = tn_per + fp_per
        pos_denom = tp_per + fn_per
        specificity_per = np.divide(
            tn_per,
            neg_denom,
            out=np.full_like(tn_per, np.nan, dtype=float),
            where=neg_denom != 0
        )
        fpr_per = np.divide(
            fp_per,
            neg_denom,
            out=np.full_like(fp_per, np.nan, dtype=float),
            where=neg_denom != 0
        )
        fnr_per = np.divide(
            fn_per,
            pos_denom,
            out=np.full_like(fn_per, np.nan, dtype=float),
            where=pos_denom != 0
        )

		# === 新增 Jaccard 计算 ===
        
        # (A) 整体 Jaccard (Micro-average)
        # 将所有预测值和真实值展平后计算 IoU，衡量全局的一致性
        metric2score["jaccard_samples"] = jaccard_score(y_true, y_pred_binary, average='samples', zero_division=1)



        # # (B) 每种疾病的 Jaccard (Per-class)
        # # average=None 会返回一个数组，形状为 (num_classes,)
        # jaccard_per_class = jaccard_score(y_true, y_pred_binary, average=None, zero_division=1)
        
        # # 将数组拆解存入字典，以便后续代码(如 Tensorboard)能正常记录标量
        # for i, score in enumerate(jaccard_per_class):
        #     metric2score[f"jaccard_class_{i}"] = score



        # 循环存入字典
        num_classes = y_true.shape[1]
        auc_per_class = []
        auprc_per_class = []
        for i in range(num_classes):
            metric2score[f"threshold_class_{i}"] = thresholds[i]
            metric2score[f"jaccard_class_{i}"] = jaccard_per[i]
            metric2score[f"precision_class_{i}"] = precision_per[i]
            metric2score[f"recall_class_{i}"] = recall_per[i]
            metric2score[f"f1_class_{i}"] = f1_per[i]
            metric2score[f"acc_class_{i}"] = acc_per[i]
            metric2score[f"specificity_class_{i}"] = specificity_per[i]
            metric2score[f"fpr_class_{i}"] = fpr_per[i]
            metric2score[f"fnr_class_{i}"] = fnr_per[i]
            
            # 单独计算 AUROC (防止某一列全为0导致整体报错)
            try:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            except ValueError:
                auc = float("nan") # AUROC is undefined when this class has only positives or only negatives.
            metric2score[f"auroc_class_{i}"] = auc
            auc_per_class.append(auc)

            # AUPRC requires at least one positive example. Use the continuous
            # probability scores so that every possible decision threshold is assessed.
            if np.any(y_true[:, i] == 1):
                auprc = average_precision_score(y_true[:, i], y_prob[:, i])
            else:
                auprc = float("nan")
            metric2score[f"auprc_class_{i}"] = auprc
            auprc_per_class.append(auprc)

            # # === 计算风险分级指标 (Prevalence, Lift, OR, NNS) ===
            # risk_metrics = self.calculate_risk_metrics(y_true[:, i], y_prob[:, i])
            # for k, v in risk_metrics.items():
            #     metric2score[f"{k}_class_{i}"] = v
            # # ========================================================
            
        valid_auc_per_class = [auc for auc in auc_per_class if not np.isnan(auc)]
        metric2score["auroc_macro"] = float(np.mean(valid_auc_per_class)) if valid_auc_per_class else float("nan")
        valid_auprc_per_class = [
            auprc for auprc in auprc_per_class if not np.isnan(auprc)
        ]
        metric2score["auprc_macro"] = (
            float(np.mean(valid_auprc_per_class))
            if valid_auprc_per_class else float("nan")
        )
        # 宏平均 Recall / F1：先分别计算每个疾病，再对所有疾病取平均。
        # 这样每个疾病权重相同，更适合观察多疾病任务中少数疾病是否被忽略。
        metric2score["precision_macro"] = float(np.mean(precision_per))
        metric2score["recall_macro"] = float(np.mean(recall_per))
        metric2score["f1_macro"] = float(np.mean(f1_per))
        metric2score["specificity_macro"] = float(np.nanmean(specificity_per)) if not np.all(np.isnan(specificity_per)) else float("nan")
        metric2score["fpr_macro"] = float(np.nanmean(fpr_per)) if not np.all(np.isnan(fpr_per)) else float("nan")
        metric2score["fnr_macro"] = float(np.nanmean(fnr_per)) if not np.all(np.isnan(fnr_per)) else float("nan")

        # 每个疾病同时考虑排序、阳性识别、综合分类和阴性识别能力。
        # 某折中指标不可定义时，只在该疾病已有的有效指标间重新归一化权重。
        disease_score_weights = np.asarray([0.30, 0.30, 0.20, 0.20], dtype=float)
        disease_scores = []
        for i, auc in enumerate(auc_per_class):
            values = np.asarray(
                [auc, f1_per[i], recall_per[i], specificity_per[i]],
                dtype=float,
            )
            valid_mask = np.isfinite(values)
            if np.any(valid_mask):
                score = float(
                    np.average(values[valid_mask], weights=disease_score_weights[valid_mask])
                )
            else:
                score = float("nan")
            disease_scores.append(score)
            metric2score[f"disease_score_class_{i}"] = score

        valid_disease_scores = np.asarray(
            [score for score in disease_scores if np.isfinite(score)],
            dtype=float,
        )
        if valid_disease_scores.size > 0:
            disease_score_macro = float(np.mean(valid_disease_scores))
            disease_score_min = float(np.min(valid_disease_scores))
            balanced_macro_score = 0.75 * disease_score_macro + 0.25 * disease_score_min
        else:
            disease_score_macro = float("nan")
            disease_score_min = float("nan")
            balanced_macro_score = float("nan")

        metric2score["disease_score_macro"] = disease_score_macro
        metric2score["disease_score_min"] = disease_score_min
        metric2score["balanced_macro_score"] = balanced_macro_score

        # =======================

        # =======================================================
        # [核心逻辑] 冠脉疾病分层筛查 (Unified Noisy-OR)
        # =======================================================

        # Noisy-OR 函数: P_union = 1 - Product(1 - P_i)
        def calc_noisy_or(probs):
            if probs.ndim == 1: return probs
            safe_prob = 1 - probs
            all_safe = np.prod(safe_prob, axis=1)
            return 1 - all_safe

        # --- A. 索引定义 ---
        # 0: 冠状动脉粥样硬化 (AS, 狭窄 < 50%)
        # 1: 冠心病 (CHD, 狭窄 >= 50%)
        # ACS (2-4), CCS (5-7) 均属于 CHD
        idx_as = [0]             
        idx_chd_clinical = [1]   
        idx_acs = [2, 3, 4]      
        idx_ccs = [5, 6, 7]      

        # --- B. 计算子综合征风险 (ACS / CCS) ---
        y_prob_acs = calc_noisy_or(y_prob[:, idx_acs])
        y_prob_ccs = calc_noisy_or(y_prob[:, idx_ccs])
        
        y_true_acs = (np.sum(y_true[:, idx_acs], axis=1) > 0).astype(int)
        y_true_ccs = (np.sum(y_true[:, idx_ccs], axis=1) > 0).astype(int)

        self._calc_and_store_group_metrics(metric2score, "ACS", y_true_acs, y_prob_acs)
        self._calc_and_store_group_metrics(metric2score, "CCS", y_true_ccs, y_prob_ccs)

        # # --- C. [Level 1] 冠心病确诊筛查 (CHD_Diagnosis) ---
        # # 逻辑：聚合 [Index 1] + [ACS] + [CCS]。严格排除 AS (Index 0)。
        # # 含义：狭窄程度 >= 50%，已形成临床冠心病。
        
        # probs_chd_components = np.column_stack([
        #     y_prob[:, idx_chd_clinical], 
        #     y_prob_acs, 
        #     y_prob_ccs
        # ])
        # y_prob_chd_total = calc_noisy_or(probs_chd_components)

        # # [关键修改] 保存患者级分数
        # patient_scores["CHD_Diagnosis_prob"] = y_prob_chd_total
        
        # # 真实标签：排除 AS
        # idx_chd_all = idx_chd_clinical + idx_acs + idx_ccs
        # y_true_chd_total = (np.sum(y_true[:, idx_chd_all], axis=1) > 0).astype(int)

        # self._calc_and_store_group_metrics(metric2score, "CHD_Diagnosis", y_true_chd_total, y_prob_chd_total)

        # --- C. [Level 1] 冠心病确诊筛查 (CHD_Diagnosis) ---
        # 修正：由于 CHD(idx 1), ACS, CCS 存在父子包含关系，使用 Noisy-OR 会导致严重膨胀。
        # 改用 Max Pooling (最大值)，取所有相关标签中最确信的一个。
        
        # 1. 收集所有相关的单病种概率
        # 包括: 冠心病父类(1) + ACS所有子类(2,3,4) + CCS所有子类(5,6,7)
        idx_chd_all_components = idx_chd_clinical + idx_acs + idx_ccs
        probs_chd_all = y_prob[:, idx_chd_all_components]
        
        # 2. 取最大值 (消除层级冗余)
        y_prob_chd_total = np.max(probs_chd_all, axis=1)
        
        # 保存分数
        patient_scores["CHD_Diagnosis_prob"] = y_prob_chd_total
        
        # 真实标签
        y_true_chd_total = (np.sum(y_true[:, idx_chd_all_components], axis=1) > 0).astype(int)
        self._calc_and_store_group_metrics(metric2score, "CHD_Diagnosis", y_true_chd_total, y_prob_chd_total)

        # --- D. [Level 2] 冠脉血管异常综合筛查 (Coronary_Abnormal) ---
        # 逻辑：AS (<50%) 和 CHD (>=50%) 的概率并集。
        # 含义：狭窄程度 > 0%，血管存在异常（包含早期硬化和晚期心脏病）。
        
        y_prob_as = y_prob[:, idx_as].flatten()
        
        # 将 [AS概率] 和 [CHD确诊概率] 进行并集
        # 公式: P_Abnormal = 1 - (1 - P_AS) * (1 - P_CHD)
        probs_broad_components = np.column_stack([
            y_prob_as,           # 阶段 1: < 50%
            y_prob_chd_total     # 阶段 2: >= 50%
        ])
        y_prob_broad = calc_noisy_or(probs_broad_components)

        # [关键修改] 保存患者级分数
        patient_scores["Coronary_Abnormal_prob"] = y_prob_broad
        
        # 真实标签：包含 AS (只要 0~7 任意一个为 1，说明血管都有问题)
        y_true_broad = (np.sum(y_true, axis=1) > 0).astype(int)
        
        self._calc_and_store_group_metrics(metric2score, "Coronary_Abnormal", y_true_broad, y_prob_broad)

        # return metric2score, dict()
        return metric2score, patient_scores

    def _calc_and_store_group_metrics(self, metric_dict, group_name, y_true, y_prob):
        """辅助函数：计算并存储聚合组指标"""
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        metric_dict[f"auroc_{group_name}"] = auc
        
        # 仅对这些聚合组计算详细的风险指标
        risk_metrics = self.calculate_risk_metrics(y_true, y_prob)
        for k, v in risk_metrics.items():
            metric_dict[f"{k}_{group_name}"] = v


    def calculate_risk_metrics(self, y_true_col, y_prob_col):
        """计算单列数据的风险分级指标"""
        metrics = {}
        
        # 1. 划分风险等级
        # 0: Low, 1: Medium, 2: High
        risk_level = np.full(y_prob_col.shape, 1) # 默认为 Medium (1)
        risk_level[y_prob_col < self.t_low] = 0   # Low (0)
        risk_level[y_prob_col >= self.t_high] = 2 # High (2)
        
        # 2. 计算各组真实患病率 (Prevalence)
        # 防止分母为 0，加一个小 epsilon
        eps = 1e-6
        
        # 总体患病率
        prev_overall = np.mean(y_true_col)
        
        # 分组计算 (0, 1, 2)
        prevalences = {}
        for level in [0, 1, 2]:
            mask = (risk_level == level)
            count = np.sum(mask)
            if count > 0:
                prev = np.mean(y_true_col[mask])
            else:
                prev = 0.0
            prevalences[level] = prev
            
        # 3. 计算核心临床指标
        prev_high = prevalences[2]
        prev_mid  = prevalences[1]
        prev_low = prevalences[0]
        
        # (A) Prevalence (High & Low) - 用于展示分层效果
        metrics["prev_high"] = prev_high
        metrics["prev_mid"] = prev_mid
        metrics["prev_low"] = prev_low
        
        # (B) Lift (提升度): High / Overall
        # 含义：高风险组的患病率是人群平均水平的多少倍
        metrics["lift"] = prev_high / (prev_overall + eps)
        
        # (C) Odds Ratio (优势比): Odds_High / Odds_Low
        # 含义：高风险组患病赔率是低风险组的多少倍
        odds_high = prev_high / (1 - prev_high + eps)
        odds_low = prev_low / (1 - prev_low + eps)
        metrics["or"] = odds_high / (odds_low + eps)
        
        # (D) NNS (需筛查人数): 1 / Prev_High
        # 含义：筛查多少个高风险人才能确诊 1 例
        metrics["nns"] = 1.0 / (prev_high + eps)
        
        return metrics
