import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import numpy as np
import torch
import os
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from plato.models.ggmlp import GGMLP, MLP
from plato.load.load_pdr_sub import PDRSubdatasetProvider
from plato.load.load_pdr import PDRDatasetProvider
from plato.baseline.pipeline_utils import SPLITS, get_device, set_drug_representation, get_training_type2split2source2dataset_or_loader, print_dataset_sizes, train_loop, save_results, set_random_seeds
from plato.baseline.pt2json import convert_and_export
# from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils import subgraph

# def rename_jaccard_keys_in_dict(results_dict, disease_names):
#     """
#     遍历结果字典，将 jaccard_class_{i} 替换为具体的疾病名称
#     """
#     # 检查基本结构
#     if 'finetune' not in results_dict: return
#     if 'source2split2metric2score' not in results_dict['finetune']: return

#     # 获取核心数据结构
#     # 结构: results_dict['finetune']['source2split2metric2score'][sourceint][split] -> {metric: score}
#     s2s2m = results_dict['finetune']['source2split2metric2score']

#     for source, split2metrics in s2s2m.items():
#         for split, metrics in split2metrics.items():
#             # 遍历所有指标键，找到需要替换的
#             # 使用 list(metrics.keys()) 创建副本，因为我们在迭代中修改字典
#             for key in list(metrics.keys()):
#                 if key.startswith('jaccard_class_'):
#                     try:
#                         # 解析索引: jaccard_class_0 -> 0
#                         idx = int(key.split('_')[-1])
#                         # 确保索引在名字列表范围内
#                         if 0 <= idx < len(disease_names):
#                             new_key = disease_names[idx]
#                             # 【核心操作】替换键名：赋值新键，弹出旧键
#                             metrics[new_key] = metrics.pop(key)
#                     except (ValueError, IndexError):
#                         pass

def rename_jaccard_keys_in_dict(results_dict, disease_names):
    """
    遍历结果字典，将 *_class_{i} 替换为 {Metric}_{疾病名称}
    例如: f1_class_0 -> F1_冠状动脉粥样硬化
    """
    if 'finetune' not in results_dict: return
    s2s2m = results_dict['finetune'].get('source2split2metric2score', {})

    # 需要处理的指标前缀
    prefixes = [
        'jaccard_class_', 'precision_class_', 'recall_class_', 'f1_class_', 'auroc_class_', 'auprc_class_', 'acc_class_',
        'specificity_class_', 'fpr_class_', 'fnr_class_'
    ]
    metric_name_map = {
        'jaccard': 'Jaccard',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1',
        'auroc': 'Auroc',
        'auprc': 'Auprc',
        'acc': 'Acc',
        'specificity': 'Specificity',
        'fpr': 'FPR',
        'fnr': 'FNR',
    }
    # === 修改 1: 增加新的指标前缀 ===
    # prefixes = [
    #     'jaccard_class_', 'precision_class_', 'recall_class_', 'f1_class_', 'auroc_class_', 'acc_class_',
    #     'prev_high_class_', 'prev_mid_class_', 'prev_low_class_', 'lift_class_', 'or_class_', 'nns_class_'
    # ]
    # ==============================

    for source, split2metrics in s2s2m.items():
        for split, metrics in split2metrics.items():
            # 使用 list() 创建副本以便在遍历中修改字典
            for key in list(metrics.keys()):
                for prefix in prefixes:
                    if key.startswith(prefix):
                        try:
                            # 提取索引: jaccard_class_0 -> 0
                            idx_str = key.split(prefix)[1] # 获取后缀数字
                            # 有些情况可能是 'jaccard_class_0' 也有可能是其他，确保分割正确
                            if not idx_str.isdigit(): continue
                            
                            idx = int(idx_str)
                            
                            if 0 <= idx < len(disease_names):
                                # 构造新键名: MetricName_DiseaseName
                                metric_key = prefix.split('_class_')[0]
                                metric_name = metric_name_map.get(metric_key, metric_key.capitalize()) # jaccard -> Jaccard
                                disease = disease_names[idx]
                                new_key = f"{metric_name}_{disease}"
                                
                                # 替换
                                metrics[new_key] = metrics.pop(key)
                                break # 匹配到一个前缀就跳出内层循环
                        except (ValueError, IndexError):
                            pass

def parse_m_layer_list(m_layer_list):
    m_layer_list_split = m_layer_list.split(",")
    if m_layer_list_split != ['']:
        m_layer_list_split = [int(i) for i in m_layer_list_split]
    else:
        m_layer_list_split = []
    return m_layer_list_split

def update_args(args):
    args.mlp_m_layer_list = parse_m_layer_list(args.mlp_m_layer_list)

    # === 新增：如果是自定义数据，直接返回，跳过后续硬编码逻辑 ===
    if args.dataset_name == 'CUSTOM':
        args.pretrain_sourceint_list = [0] # 假设您的数据来源ID是0
        args.finetune_sourceint_list = [0]
        args.mode = "custom"
        # 多疾病任务未显式指定选模指标时，默认使用兼顾平均表现和最差疾病的综合分数。
        if args.selection_metric == "neg_mse":
            args.selection_metric = "balanced_macro_score"
        return args
    # ========================================================

    # Set subtype arguments
    args.mode = "subtype"
    if args.dataset_name in ["BC", "CH", "ME", "NSCLC", "SCLC"]:
        args.pretrain_sourceint_list = [0]
        args.finetune_sourceint_list = [0]
        args.subtype_category = "cancer-type"
    elif args.dataset_name in ["CRC", "PDAC", "BRCA", "MNSCLC", "CM"]:
        args.pretrain_sourceint_list = [1]
        args.finetune_sourceint_list = [1]
        args.subtype_category = "Tumor Type"
    else:
        assert(False)
    dataset_name2subtype_name = {"BC": "Breast Carcinoma", "CH": "Chondrosarcoma", "ME": "Melanoma", "NSCLC": "Non-Small Cell Lung Carcinoma", "SCLC": "Small Cell Lung Carcinoma", "CRC": "CRC", "PDAC": "PDAC", "BRCA": "BRCA", "MNSCLC": "NSCLC", "CM": "CM"}
    args.subtype_name = dataset_name2subtype_name[args.dataset_name]
    return args

def validate_args(args):
    if args.model == "GGMLP":
        assert(not(args.drugkg))
    if args.mode == "subtype":
        assert(args.pretrain_sourceint_list == args.finetune_sourceint_list)
    if args.use_5fold_cv:
        assert(args.cv_folds >= 2)
        assert(args.cv_inner_val_folds >= 2)
    if args.use_disease_threshold_optimization:
        assert(0.0 <= args.threshold_opt_min < args.threshold_opt_max <= 1.0)
        assert(args.threshold_opt_step > 0)
        assert(0.0 <= args.threshold_opt_fallback <= 1.0)
        assert(args.threshold_opt_min_pos >= 1)
        assert(args.threshold_opt_min_neg >= 1)
        assert(args.threshold_typed_constraint_penalty >= 0)
        assert(args.threshold_typed_distance_penalty >= 0)
        threshold_score_weights = [
            args.threshold_score_f1_weight,
            args.threshold_score_recall_weight,
            args.threshold_score_specificity_weight,
            args.threshold_score_precision_weight,
        ]
        assert(all(weight >= 0 for weight in threshold_score_weights))
        assert(sum(threshold_score_weights) > 0)

def cleanup_default_results_dir(args):
    """当输出文件是默认 results/result.pt 时，先清空旧 results 目录。"""
    filename = os.path.normpath(args.filename)
    default_filename = os.path.normpath("results/result.pt")
    if filename != default_filename:
        return

    results_dir = os.path.abspath(os.path.dirname(default_filename))
    expected_dir = os.path.abspath("results")
    # 安全保护：只允许删除当前运行目录下名为 results 的目录。
    if results_dir != expected_dir or os.path.basename(results_dir) != "results":
        raise RuntimeError(f"拒绝删除非预期目录: {results_dir}")

    if os.path.isdir(results_dir):
        print(f"[INFO] 检测到 --filename 为 {args.filename}，正在删除旧 results 目录: {results_dir}")
        shutil.rmtree(results_dir)

def make_multilabel_stratified_folds(labels, n_splits, seed):
    """近似多标签分层划分，同时考虑每个疾病的阳性和阴性分布。"""
    labels = np.asarray(labels).astype(int)
    n_samples = labels.shape[0]
    rng = np.random.default_rng(seed)

    # 同时分层阳性和阴性，避免高阳性疾病的少量阴性集中到某一折。
    strat_labels = np.concatenate([labels, 1 - labels], axis=1)
    base_size = n_samples // n_splits
    extra = n_samples % n_splits
    target_sizes = np.array([base_size + (1 if i < extra else 0) for i in range(n_splits)], dtype=float)

    fold_indices = [[] for _ in range(n_splits)]
    fold_label_counts = np.zeros((n_splits, strat_labels.shape[1]), dtype=float)
    fold_sizes = np.zeros(n_splits, dtype=float)
    assigned = np.zeros(n_samples, dtype=bool)

    # 稀有标签优先分配：少阳性疾病和少阴性疾病都尽量分散到各折。
    label_counts = strat_labels.sum(axis=0)
    label_order = np.argsort(np.where(label_counts > 0, label_counts, n_samples + 1))

    for label_idx in label_order:
        candidates = np.where((strat_labels[:, label_idx] == 1) & (~assigned))[0]
        if len(candidates) == 0:
            continue
        rng.shuffle(candidates)

        for sample_idx in candidates:
            available_folds = np.where(fold_sizes < target_sizes)[0]
            if len(available_folds) == 0:
                break

            # 优先放入当前标签数量最少、同时容量还没满的折。
            label_score = fold_label_counts[available_folds, label_idx]
            size_score = fold_sizes[available_folds] / np.maximum(target_sizes[available_folds], 1.0)
            score = label_score + 0.01 * size_score + rng.random(len(available_folds)) * 1e-6
            best_fold = int(available_folds[np.argmin(score)])

            fold_indices[best_fold].append(int(sample_idx))
            fold_label_counts[best_fold] += strat_labels[sample_idx]
            fold_sizes[best_fold] += 1
            assigned[sample_idx] = True

    # 没有被稀有标签流程覆盖到的样本，按折大小补齐。
    remaining = np.where(~assigned)[0]
    rng.shuffle(remaining)
    for sample_idx in remaining:
        available_folds = np.where(fold_sizes < target_sizes)[0]
        best_fold = int(available_folds[np.argmin(fold_sizes[available_folds])])
        fold_indices[best_fold].append(int(sample_idx))
        fold_label_counts[best_fold] += strat_labels[sample_idx]
        fold_sizes[best_fold] += 1

    return [np.array(indices, dtype=int) for indices in fold_indices]

def build_multilabel_cv_split_dicts(provider, args):
    """构建外层 5 折测试集，并在每折训练部分内部划分验证集。"""
    labels = provider.y_dict["response"].cpu().numpy()
    all_idx = np.arange(labels.shape[0])
    outer_folds = make_multilabel_stratified_folds(labels, args.cv_folds, args.seed)
    split_dicts = []

    for fold_idx, test_idx in enumerate(outer_folds):
        temp_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        inner_labels = labels[temp_idx]
        inner_folds = make_multilabel_stratified_folds(
            inner_labels,
            args.cv_inner_val_folds,
            args.seed + 1000 + fold_idx
        )
        val_inner_idx = inner_folds[fold_idx % args.cv_inner_val_folds]
        val_idx = temp_idx[val_inner_idx]
        train_idx = np.setdiff1d(temp_idx, val_idx, assume_unique=False)

        split_dicts.append({
            "fold": fold_idx,
            "train": torch.as_tensor(train_idx, dtype=torch.long),
            "val": torch.as_tensor(val_idx, dtype=torch.long),
            "test": torch.as_tensor(test_idx, dtype=torch.long),
        })

        print(
            f"[CV] fold {fold_idx + 1}/{args.cv_folds}: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    return split_dicts

def get_args():
    parser = argparse.ArgumentParser(description='Single source prediction pipeline')
    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=19999)
    parser.add_argument('--selection_metric', type = str, default = 'neg_mse') # pearsonr
    parser.add_argument('--load_only', action='store_true', default = False)
    parser.add_argument('--tensorboard_dir', default = None)
    parser.add_argument('--dataset_name', choices = ['BC', 'CH', 'ME', 'NSCLC', 'SCLC', 'CRC', 'PDAC', 'BRCA', 'MNSCLC', 'CM', 'CUSTOM'])
    parser.add_argument('--use_5fold_cv', action='store_true', default=False, help='启用多标签分层 5 折交叉验证')
    parser.add_argument('--cv_folds', type=int, default=5, help='外层交叉验证折数')
    parser.add_argument('--cv_inner_val_folds', type=int, default=5, help='每个外层训练部分内部划分验证集的折数')
    
    # Training hyperparameter arguments
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    
    # Pipeline arguments
    parser.add_argument('--sample_frac', type=float, default=1)
    parser.add_argument('--drugkg', action='store_true', default = False) # use drug kg embedding

    # KG Embedding arguments
    parser.add_argument('--kg_embed_size', type=int, default=200)
    parser.add_argument('--embedding_model', choices = ["ComplEx"], default = "ComplEx")

    # Model arguments
    # parser.add_argument('--model', choices = ["GGMLP"], default = 'GGMLP')
    parser.add_argument('--model', choices = ["GGMLP", "MLP"], default = 'GGMLP')
    parser.add_argument('--skip_connection', action='store_true', default = False) # use drug kg embedding
    parser.add_argument('--use_bn', action='store_true', default = False) 
    parser.add_argument('--simple', action='store_true', default = False) 
    parser.add_argument('--train_meta', action='store_true', default = False) 
    parser.add_argument('--gene_nonlin', type = str, choices = ["none", "softmax", "relu", "leakyrelu", "tanh"], default = 'none')
    parser.add_argument('--drug_nonlin', type = str, choices = ["none", "softmax", "relu", "leakyrelu", "tanh"], default = 'none')
    parser.add_argument('--gene_div', type=float, default = 1.)
    parser.add_argument('--drug_div', type=float, default = 1.)
    parser.add_argument('--enlarge', type=int, default=20)
    parser.add_argument('--beta', type=float, default = 0.)
    parser.add_argument('--mp', action='store_true', default = False) 
    parser.add_argument('--scalar_attn', action='store_true', default = False) 
    parser.add_argument('--num_edges_sampled', type=int, default=50000)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--l1_weight', type=float, default = 0.0)
    parser.add_argument('--l2_weight', type=float, default = 0.0)
    parser.add_argument('--mlp_m_layer_list', type = str, default = '32,32,1')
    # === 新增：是否开启预测层规则拦截的动态开关 ===
    parser.add_argument('--use_rule_mask', action='store_true', default=False, help='Enable prediction layer rule interception')
    # === 【修改】：新增 Method C (深层图谱融合) 的命令行开关 ===
    parser.add_argument('--use_rule_graph', action='store_true', default=False, help='Enable Rule Graph Fusion: Inject disease hub nodes and use bidirectional edges in subgraph')
    # === 【新增】：自动计算类别平衡权重的命令行开关 ===
    parser.add_argument('--use_class_weights', action='store_true', default=False, help='Enable automatic calculation and application of class balance weights (pos_weight)')
    # 疾病级阈值优化：训练结束后只用验证集为每个疾病单独选择判阳阈值，测试集不参与选阈值。
    parser.add_argument('--use_disease_threshold_optimization', action='store_true', default=False, help='启用疾病级判阳阈值优化')
    parser.add_argument('--threshold_opt_min', type=float, default=0.10, help='疾病级阈值搜索下限')
    parser.add_argument('--threshold_opt_max', type=float, default=0.90, help='疾病级阈值搜索上限')
    parser.add_argument('--threshold_opt_step', type=float, default=0.05, help='疾病级阈值搜索步长')
    parser.add_argument('--threshold_opt_fallback', type=float, default=0.5, help='验证集样本不足时回退使用的阈值')
    parser.add_argument('--threshold_opt_min_pos', type=int, default=2, help='验证集中执行阈值优化所需的最少阳性样本数')
    parser.add_argument('--threshold_opt_min_neg', type=int, default=2, help='验证集中执行阈值优化所需的最少阴性样本数')
    parser.add_argument('--threshold_opt_min_recall', type=float, default=0.0, help='阈值搜索时允许的最小 Recall 约束')
    parser.add_argument('--threshold_opt_min_specificity', type=float, default=0.0, help='阈值搜索时允许的最小 Specificity 约束')
    # 反馈驱动的难阴性采样，与类别权重互补：类别权重保护少数阳性，
    # 该采样器则让下一轮训练更关注当前模型容易混淆的阴性病例。
    parser.add_argument('--use_adaptive_negative_sampling', action='store_true', default=False, help='启用多标签训练中的反馈驱动难阴性采样')
    parser.add_argument('--adaptive_neg_warmup_epochs', type=int, default=1, help='开始更新难阴性采样权重前的预热轮数')
    parser.add_argument('--adaptive_neg_update_freq', type=int, default=1, help='预热后每隔多少轮更新一次难阴性采样权重')
    parser.add_argument('--adaptive_neg_epoch_multiplier', type=float, default=1.0, help='每轮采样样本数相对训练集大小的倍数')
    parser.add_argument('--adaptive_neg_boundary_threshold', type=float, default=0.2, help='边界阴性样本的预测概率下限')
    parser.add_argument('--adaptive_neg_hard_threshold', type=float, default=0.5, help='难阴性样本的预测概率阈值')
    parser.add_argument('--adaptive_neg_boundary_bonus', type=float, default=1.0, help='边界阴性样本的采样权重增量')
    parser.add_argument('--adaptive_neg_hard_bonus', type=float, default=2.0, help='难阴性样本的采样权重增量')
    parser.add_argument('--adaptive_neg_tau_fp', type=float, default=0.15, help='提高难阴性采样强度的假阳性率阈值')
    parser.add_argument('--adaptive_neg_tau_fn', type=float, default=0.20, help='降低难阴性采样强度的假阴性率阈值')
    parser.add_argument('--adaptive_neg_tau_recall', type=float, default=0.75, help='决定是否提高阴性采样强度的召回率下限')
    parser.add_argument('--adaptive_neg_alpha_min', type=float, default=0.5, help='单个标签阴性采样强度下限')
    parser.add_argument('--adaptive_neg_alpha_max', type=float, default=3.0, help='单个标签阴性采样强度上限')
    parser.add_argument('--adaptive_neg_growth', type=float, default=1.2, help='假阳性率偏高时采样强度的乘性增长系数')
    parser.add_argument('--adaptive_neg_decay', type=float, default=0.8, help='假阴性率偏高时采样强度的乘性衰减系数')
    # 双向自适应采样：按疾病分别根据漏诊和误报反馈增强阳性或阴性难样本。
    parser.add_argument('--use_bidirectional_adaptive_sampling', action='store_true', default=False, help='启用双向自适应采样')
    parser.add_argument('--use_disease_typed_bidirectional_sampling', action='store_true', default=False, help='启用疾病分型驱动的双向采样参数自适应机制')
    parser.add_argument('--adaptive_bidir_warmup_epochs', type=int, default=1, help='开始更新双向采样权重前的预热轮数')
    parser.add_argument('--adaptive_bidir_update_freq', type=int, default=1, help='预热后每隔多少轮更新一次双向采样权重')
    parser.add_argument('--adaptive_bidir_epoch_multiplier', type=float, default=1.0, help='每轮采样样本数相对训练集大小的倍数')
    parser.add_argument('--adaptive_bidir_pos_hard_threshold', type=float, default=0.5, help='难阳性样本的预测概率上限')
    parser.add_argument('--adaptive_bidir_pos_boundary_threshold', type=float, default=0.8, help='边界阳性样本的预测概率上限')
    parser.add_argument('--adaptive_bidir_neg_boundary_threshold', type=float, default=0.35, help='边界阴性样本的预测概率下限')
    parser.add_argument('--adaptive_bidir_neg_hard_threshold', type=float, default=0.6, help='难阴性样本的预测概率下限')
    parser.add_argument('--adaptive_bidir_pos_boundary_bonus', type=float, default=1.0, help='边界阳性样本的采样权重增量')
    parser.add_argument('--adaptive_bidir_pos_hard_bonus', type=float, default=2.0, help='难阳性样本的采样权重增量')
    parser.add_argument('--adaptive_bidir_neg_boundary_bonus', type=float, default=0.5, help='边界阴性样本的采样权重增量')
    parser.add_argument('--adaptive_bidir_neg_hard_bonus', type=float, default=1.2, help='难阴性样本的采样权重增量')
    parser.add_argument('--adaptive_bidir_tau_fnr', type=float, default=0.35, help='提高阳性采样强度的假阴性率阈值')
    parser.add_argument('--adaptive_bidir_tau_fpr', type=float, default=0.45, help='提高阴性采样强度的假阳性率阈值')
    parser.add_argument('--adaptive_bidir_tau_recall', type=float, default=0.55, help='提高阳性采样强度的召回率下限')
    parser.add_argument('--adaptive_bidir_tau_specificity', type=float, default=0.45, help='提高阴性采样强度的特异度下限')
    parser.add_argument('--adaptive_bidir_alpha_min', type=float, default=0.5, help='单个标签采样强度下限')
    parser.add_argument('--adaptive_bidir_alpha_max', type=float, default=3.0, help='单个标签采样强度上限')
    parser.add_argument('--adaptive_bidir_pos_alpha_max', type=float, default=3.0, help='阳性采样强度上限')
    parser.add_argument('--adaptive_bidir_neg_alpha_max', type=float, default=2.5, help='阴性采样强度上限')
    parser.add_argument('--adaptive_bidir_growth', type=float, default=1.08, help='单侧错误偏高时采样强度的乘性增长系数')
    parser.add_argument('--adaptive_bidir_both_growth', type=float, default=1.05, help='正负两侧错误都高时采样强度的小幅增长系数')
    parser.add_argument('--adaptive_bidir_decay', type=float, default=0.95, help='相反方向或稳定标签的采样强度衰减系数')
    parser.add_argument('--adaptive_bidir_low_prevalence_threshold', type=float, default=0.25, help='低阳性占比疾病阈值')
    parser.add_argument('--adaptive_bidir_high_prevalence_threshold', type=float, default=0.65, help='高阳性占比疾病阈值')
    parser.add_argument('--adaptive_bidir_prevalence_boost', type=float, default=1.15, help='阳性占比先验对应方向的增强系数')
    parser.add_argument('--adaptive_bidir_opposite_prevalence_scale', type=float, default=0.75, help='阳性占比先验相反方向的增长缩放系数')
    parser.add_argument('--adaptive_bidir_pressure_scale', type=float, default=1.0, help='FNR/FPR 压力分数缩放系数')
    parser.add_argument('--adaptive_bidir_balance_margin', type=float, default=0.15, help='判断正负错误压力差异的安全边界')
    parser.add_argument('--adaptive_bidir_high_fpr_guard', type=float, default=0.75, help='假阳性率过高时的强制阴性补偿阈值')
    parser.add_argument('--adaptive_bidir_high_fnr_guard', type=float, default=0.75, help='假阴性率过高时的强制阳性补偿阈值')
    parser.add_argument('--adaptive_bidir_guard_growth', type=float, default=1.15, help='触发强制纠偏时对应方向的增长系数')
    parser.add_argument('--adaptive_bidir_guard_decay', type=float, default=0.8, help='触发强制纠偏时相反方向的衰减系数')
    parser.add_argument('--adaptive_bidir_overfit_gap', type=float, default=0.25, help='训练集与验证集指标差距超过该值时触发过拟合保护')
    parser.add_argument('--adaptive_bidir_overfit_decay', type=float, default=0.85, help='触发过拟合保护时对应采样强度的衰减系数')
    parser.add_argument('--adaptive_bidir_sample_bonus_max', type=float, default=3.0, help='单个样本双向采样权重增量上限')
    parser.add_argument('--disable_adaptive_bidir_prevalence_prior', action='store_true', default=False, help='关闭双向采样中的阳性占比先验')
    parser.add_argument('--disease_typed_very_low_prevalence_threshold', type=float, default=0.10, help='疾病分型中极低阳性占比阈值')
    parser.add_argument('--disease_typed_very_high_prevalence_threshold', type=float, default=0.90, help='疾病分型中极高阳性占比阈值')
    parser.add_argument('--disease_typed_low_prev_pos_alpha_max', type=float, default=3.5, help='少阳性漏诊型疾病的阳性采样强度上限')
    parser.add_argument('--disease_typed_low_prev_neg_alpha_max', type=float, default=1.4, help='少阳性漏诊型疾病的阴性采样强度上限')
    parser.add_argument('--disease_typed_very_low_prev_pos_alpha_max', type=float, default=4.0, help='极少阳性疾病的阳性采样强度上限')
    parser.add_argument('--disease_typed_very_low_prev_neg_alpha_max', type=float, default=1.2, help='极少阳性疾病的阴性采样强度上限')
    parser.add_argument('--disease_typed_high_prev_neg_alpha_max', type=float, default=3.2, help='高阳性误报型疾病的阴性采样强度上限')
    parser.add_argument('--disease_typed_high_prev_pos_alpha_max', type=float, default=1.6, help='高阳性误报型疾病的阳性采样强度上限')
    parser.add_argument('--disease_typed_very_high_prev_neg_alpha_cap', type=float, default=2.6, help='极高阳性疾病的阴性采样强度保护上限')
    parser.add_argument('--disease_typed_confusion_alpha_max', type=float, default=2.2, help='双向混淆型疾病的双侧采样强度上限')
    parser.add_argument('--disease_typed_pos_growth_boost', type=float, default=1.2, help='疾病分型机制中阳性增强增长系数放大倍数')
    parser.add_argument('--disease_typed_neg_growth_boost', type=float, default=1.2, help='疾病分型机制中阴性增强增长系数放大倍数')
    parser.add_argument('--disease_typed_low_prev_neg_growth_scale', type=float, default=0.45, help='少阳性疾病阴性增强增长缩放系数')
    parser.add_argument('--disease_typed_high_prev_pos_growth_scale', type=float, default=0.55, help='高阳性疾病阳性增强增长缩放系数')
    parser.add_argument('--disease_typed_low_prev_neg_bonus_scale', type=float, default=0.5, help='少阳性疾病阴性难样本 bonus 缩放系数')
    parser.add_argument('--disease_typed_high_prev_neg_bonus_boost', type=float, default=1.2, help='高阳性疾病阴性难样本 bonus 放大倍数')
    # 稳定疾病保护：已经在验证集表现稳定的疾病，自动减少或关闭额外采样。
    parser.add_argument('--disease_typed_stable_auroc_threshold', type=float, default=0.85, help='稳定疾病保护的 AUROC 下限')
    parser.add_argument('--disease_typed_stable_f1_threshold', type=float, default=0.60, help='稳定疾病保护的 F1 下限')
    parser.add_argument('--disease_typed_stable_recall_threshold', type=float, default=0.70, help='稳定疾病保护的 Recall 下限')
    parser.add_argument('--disease_typed_stable_specificity_threshold', type=float, default=0.70, help='稳定疾病保护的 Specificity 下限')
    parser.add_argument('--disease_typed_stable_min_pos', type=int, default=3, help='触发稳定疾病保护所需的最少阳性样本数')
    parser.add_argument('--disease_typed_stable_min_neg', type=int, default=3, help='触发稳定疾病保护所需的最少阴性样本数')
    parser.add_argument('--disease_typed_stable_decay', type=float, default=0.5, help='稳定疾病采样强度回退到 1 的衰减系数')
    # 疾病分型阈值优化：按疾病类型动态设置 Recall/Specificity 软约束。
    parser.add_argument('--threshold_typed_low_prev_min_recall', type=float, default=0.40, help='少阳性漏诊型疾病的阈值优化 Recall 下限')
    parser.add_argument('--threshold_typed_low_prev_min_specificity', type=float, default=0.20, help='少阳性漏诊型疾病的阈值优化 Specificity 下限')
    parser.add_argument('--threshold_typed_high_prev_min_recall', type=float, default=0.60, help='高阳性误报型疾病的阈值优化 Recall 下限')
    parser.add_argument('--threshold_typed_high_prev_min_specificity', type=float, default=0.20, help='高阳性误报型疾病的阈值优化 Specificity 下限')
    parser.add_argument('--threshold_typed_stable_min_recall', type=float, default=0.60, help='稳定型疾病阈值优化 Recall 下限')
    parser.add_argument('--threshold_typed_stable_min_specificity', type=float, default=0.60, help='稳定型疾病阈值优化 Specificity 下限')
    parser.add_argument('--threshold_typed_confusion_min_recall', type=float, default=0.35, help='双向混淆型疾病阈值优化 Recall 下限')
    parser.add_argument('--threshold_typed_confusion_min_specificity', type=float, default=0.35, help='双向混淆型疾病阈值优化 Specificity 下限')
    parser.add_argument('--threshold_typed_constraint_penalty', type=float, default=2.0, help='阈值优化中违反 Recall/Specificity 约束的惩罚强度')
    parser.add_argument('--threshold_typed_distance_penalty', type=float, default=0.05, help='阈值远离默认 0.5 时的轻微惩罚')
    # 疾病级阈值优化的多指标综合目标，默认权重之和为 1。
    parser.add_argument('--threshold_score_f1_weight', type=float, default=0.40, help='阈值综合得分中的 F1 权重')
    parser.add_argument('--threshold_score_recall_weight', type=float, default=0.25, help='阈值综合得分中的 Recall 权重')
    parser.add_argument('--threshold_score_specificity_weight', type=float, default=0.20, help='阈值综合得分中的 Specificity 权重')
    parser.add_argument('--threshold_score_precision_weight', type=float, default=0.15, help='阈值综合得分中的 Precision 权重')
    # Output
    args = parser.parse_args()
    args = update_args(args)
    print(args)
    return args
        
if __name__ == "__main__":
    # Set up
    args = get_args()
    validate_args(args)
    cleanup_default_results_dir(args)
    device = get_device(args.device)
    assert(type(args.pretrain_sourceint_list) is list)
    assert(type(args.finetune_sourceint_list) is list)

    # Set seed
    set_random_seeds(args.seed)

    if os.path.exists(args.filename):
        print(f"************************{args.filename} already exists!************************")
    else:
        # Load Data
        if args.mode == "subtype":
            assert(len(args.finetune_sourceint_list) == 1)
            subtype_source = {0: "cell", 1: "mouse", 2: "patient"}[args.finetune_sourceint_list[0]]
            args.subtype_response_col = {"cell": "ln-ic50", "mouse": "min-avg-pct-tumor-growth", "patient": "PFI.time"}[subtype_source]
            provider = PDRSubdatasetProvider(subtype_source, args.subtype_category, args.subtype_name, args.subtype_response_col, load_dir=args.load_dir, cache_dir=args.cache_dir)
            provider.source2task = {"cell": "numeric", "mouse": "numeric", "patient": "numeric"}
        
        # === 新增：自定义数据加载逻辑 ===
        elif args.dataset_name == 'CUSTOM':
            # 指向您生成的 .pt 文件路径
            custom_file_path = os.path.join(args.cache_dir, "my_multilabel_data.pt")
            print(f"Loading custom dataset from: {custom_file_path}")
            
            provider = PDRDatasetProvider(
                cache_file=custom_file_path,
                load_from_scratch=False,
                skip_kg=False,
                load_dir=args.load_dir,
                cache_dir=args.cache_dir
            )
            # 关键：告诉模型这是多标签分类任务，以便使用正确的 Loss (BCE)
            # 注意：您必须确保 pipeline_utils.py 中的 train_loop 能处理 'multilabel'
            provider.source2task = {0: "multilabel"} 
        # =================================

        else:
            assert(False)
        print(provider.source2task)

        # Split Data
        if args.use_5fold_cv:
            cv_split_dicts = build_multilabel_cv_split_dicts(provider, args)
            train_idx, val_idx, test_idx = cv_split_dicts[0]['train'], cv_split_dicts[0]['val'], cv_split_dicts[0]['test']
        else:
            split_dict = provider.get_split_idx(from_scratch = True, seed = args.seed)
            train_idx, val_idx, test_idx = split_dict['train'], split_dict['val'], split_dict['test']
            cv_split_dicts = [{"fold": 0, "train": train_idx, "val": val_idx, "test": test_idx}]

        # # Set up drug representation
        # provider = set_drug_representation(provider, args.drugkg)
        # gene_dim = provider.X_expression.size(1)
        # drug_dim = provider.y_dict["drug"].size(1)
        # print(provider.y_dict['drug'])

        # # Set up datasets for training and pre-training
        # training_type2split2source2dataset_or_loader = get_training_type2split2source2dataset_or_loader(provider, args.finetune_sourceint_list, args.pretrain_sourceint_list, train_idx, val_idx, test_idx, args.batch_size, args.sample_frac)
        # print("Printing dataset sizes...")
        # print_dataset_sizes(training_type2split2source2dataset_or_loader, args.pretrain_sourceint_list, args.finetune_sourceint_list)
        # if args.mp:
        #     gene_edge_index, _ = subgraph(torch.LongTensor(provider.kg_mapping_dict['X_expression']), provider.kg.data.edge_index, relabel_nodes=True)
        #     drug_edge_index, _ = subgraph(torch.LongTensor(provider.kg_mapping_dict['drug']), provider.kg.data.edge_index, relabel_nodes=True)
        # else:
        #     gene_edge_index = None
        #     drug_edge_index = None
        # Set up drug representation
        provider = set_drug_representation(provider, args.drugkg)
        gene_dim = provider.X_expression.size(1)
        drug_dim = provider.y_dict["drug"].size(1)
        print(provider.y_dict['drug'])

        # ====================================================================
        # === [Method C] 注入虚拟疾病枢纽节点 (Virtual Disease Hub Nodes) ===
        num_disease_nodes = 0
        if args.use_rule_graph and args.dataset_name == 'CUSTOM' and args.mp:
            DISEASE_NAMES = [
                '冠状动脉粥样硬化', '冠状动脉粥样硬化性心脏病', '不稳定型心绞痛', 
                'ST段抬高型心肌梗死', '非ST段抬高型心肌梗死', '稳定型心绞痛', 
                '隐匿性或无症状型心肌缺血', '缺血性心肌病'
            ]
            disease_kg_ids = [provider.kg.ent2id[name] for name in DISEASE_NAMES if name in provider.kg.ent2id]
            num_disease_nodes = len(disease_kg_ids)
            
            if num_disease_nodes > 0:
                print(f"\n>>> [INFO] Method C (深层图谱融合): 正在向 Subgraph 中注入 {num_disease_nodes} 个虚拟疾病枢纽！")
                # 将疾病节点的 ID 追加到特征节点列表中，强制保留它们
                augmented_ids = np.concatenate([provider.kg_mapping_dict['X_expression'], np.array(disease_kg_ids)])
                provider.kg_mapping_dict['X_expression'] = augmented_ids
        
        # 图谱上的总节点数变成了: 原始特征数 + 疾病节点数
        augmented_gene_dim = gene_dim + num_disease_nodes
        # ====================================================================

        # Set up datasets for training and pre-training
        training_type2split2source2dataset_or_loader = get_training_type2split2source2dataset_or_loader(
            provider,
            args.finetune_sourceint_list,
            args.pretrain_sourceint_list,
            train_idx,
            val_idx,
            test_idx,
            args.batch_size,
            args.sample_frac,
            foldwise_minmax=(args.dataset_name == "CUSTOM"),
        )
        print("Printing dataset sizes...")
        print_dataset_sizes(training_type2split2source2dataset_or_loader, args.pretrain_sourceint_list, args.finetune_sourceint_list)
        if args.mp:
            gene_edge_index, _ = subgraph(torch.LongTensor(provider.kg_mapping_dict['X_expression']), provider.kg.data.edge_index, relabel_nodes=True)
            # 同一头尾节点的重复记录只保留一条，避免重复边改变消息聚合权重。
            gene_edge_index = torch.unique(gene_edge_index, dim=1)
            
            # === [Method C 核心] 强制让边变成双向 (Undirected) ===
            # 因为图谱中原有的规则边是 (特征 -> 疾病)，
            # 如果不转为双向，信息就只流进疾病出不来。双向化能让枢纽的诊断逻辑回灌给特征！
            if num_disease_nodes > 0:
                gene_edge_index = torch.cat([gene_edge_index, gene_edge_index[[1, 0]]], dim=1)
            # ===================================================
                
            drug_edge_index, _ = subgraph(torch.LongTensor(provider.kg_mapping_dict['drug']), provider.kg.data.edge_index, relabel_nodes=True)
        else:
            gene_edge_index = None
            drug_edge_index = None
        
        # assert(args.model == "GGMLP")
        # model = GGMLP(m_layer_list = args.mlp_m_layer_list, node_dim = args.kg_embed_size, gene_dim = gene_dim, drug_dim = drug_dim, provider = provider, device = device, skip_connection=args.skip_connection, use_bn=args.use_bn, simple=args.simple, train_meta=args.train_meta, drug_nonlin=args.drug_nonlin, gene_nonlin=args.gene_nonlin, drug_div=args.drug_div, gene_div=args.gene_div, l1_weight=args.l1_weight, l2_weight=args.l2_weight, enlarge=args.enlarge, mp=args.mp, beta=args.beta, gene_edge_index=gene_edge_index, drug_edge_index=drug_edge_index, num_edges_sampled=args.num_edges_sampled, scalar_attn=args.scalar_attn).to(device)
        # === 模型选择逻辑 ===
        if args.model == "GGMLP":
            model = GGMLP(
                m_layer_list=args.mlp_m_layer_list, 
                node_dim=args.kg_embed_size, 
                # gene_dim=gene_dim, 
                gene_dim=augmented_gene_dim, 
                drug_dim=drug_dim, 
                provider=provider, 
                device=device, 
                skip_connection=args.skip_connection, 
                use_bn=args.use_bn, 
                simple=args.simple, 
                train_meta=args.train_meta, 
                drug_nonlin=args.drug_nonlin, 
                gene_nonlin=args.gene_nonlin, 
                drug_div=args.drug_div, 
                gene_div=args.gene_div, 
                l1_weight=args.l1_weight, 
                l2_weight=args.l2_weight, 
                enlarge=args.enlarge, 
                mp=args.mp, 
                beta=args.beta, 
                gene_edge_index=gene_edge_index, 
                drug_edge_index=drug_edge_index, 
                num_edges_sampled=args.num_edges_sampled, 
                scalar_attn=args.scalar_attn
            ).to(device)

            # === [Method C] 保存真实的特征维度，供前向传播做 Padding 用 ===
            model.original_gene_dim = gene_dim
            model.num_disease_nodes = num_disease_nodes
            # ============================================================
            
        elif args.model == "MLP":
            print("Initializing Baseline MLP Model (No Knowledge Graph)...")
            # MLP 的输入维度是 基因维度 + 药物维度 (拼接)
            input_dim = gene_dim + drug_dim
            
            model = MLP(
                input_dim=input_dim,
                m_layer_list=args.mlp_m_layer_list,
                provider=provider,
                device=device,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                use_bn=args.use_bn
            ).to(device)
        # ============================

        # ====== 新增：把拦截开关挂载到模型上 ======
        model.use_rule_mask = args.use_rule_mask
        if model.use_rule_mask:
            print("\n>>> [INFO] 预测层医学逻辑拦截 (Prediction Layer Interception) 已开启！ <<<")
        # ==========================================
        # ====== 新增：全自动定位真实特征列索引 (适配 One-Hot 编码) ======
        feat_indices = {}
        if args.dataset_name == 'CUSTOM':
            kg_ids = provider.kg_mapping_dict['X_expression']
            # 将 rules.json 中的核心特征映射为变量名
            track_dict = {
                '心电图_有心肌缺血表现': 'ECG_ISCHEMIA',
                '性质_压榨性疼痛': 'NATURE_CRUSH',
                '性质_憋闷感、濒死感': 'NATURE_SUFF',
                '硝酸甘油_3-5min缓解': 'NITRO_RELIEF',
                '大汗_是': 'SWEAT_YES',
                '双下肢无力_是': 'WEAK_LEGS',
                '持续时间_20-60min': 'DUR_LONG1',
                '持续时间_>1h': 'DUR_LONG2',
                '胸闷或及胸痛_无': 'NO_PAIN',
                '吸烟_是': 'RISK_SMOKE'
            }
            for i, kg_id in enumerate(kg_ids):
                feat_name = provider.kg.id2ent[kg_id].strip()
                if feat_name in track_dict:
                    feat_indices[track_dict[feat_name]] = i
            
            print("\n>>> [INFO] 自动识别到的关键 0/1 特征列索引:", feat_indices)
        
        # 将索引字典挂载到模型上
        model.feat_indices = feat_indices
        # ==========================================
        print(model)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Set up the tensorboard writer
        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y.%H.%M.%S")

        # Exit if just loading data
        if args.load_only:
            assert(False)

        # For each run, do pretraining and then fine-tuning
        results_dict_list = []
        for fold_split in cv_split_dicts:
            fold_idx = fold_split["fold"]
            if args.use_5fold_cv:
                print(f"\n==================== CV fold {fold_idx + 1}/{args.cv_folds} ====================")

            training_type2split2source2dataset_or_loader = get_training_type2split2source2dataset_or_loader(
                provider,
                args.finetune_sourceint_list,
                args.pretrain_sourceint_list,
                fold_split["train"],
                fold_split["val"],
                fold_split["test"],
                args.batch_size,
                args.sample_frac,
                foldwise_minmax=(args.dataset_name == "CUSTOM"),
            )
            print("Printing dataset sizes...")
            print_dataset_sizes(training_type2split2source2dataset_or_loader, args.pretrain_sourceint_list, args.finetune_sourceint_list)

            for run in range(args.runs):
                # Finish setting up the tensorboard writer
                writer_prefix = f"{args.tensorboard_dir}/{date_time}_model_{args.model}"
                if args.use_5fold_cv:
                    writer_prefix = f"{writer_prefix}_fold_{fold_idx + 1}"
                split2writer = {"train": SummaryWriter(f"{writer_prefix}_run_{run}.train"), "val": SummaryWriter(f"{writer_prefix}_run_{run}.val"), "test": SummaryWriter(f"{writer_prefix}_run_{run}.test")}

                print(f'==============fold{fold_idx}_run{run}')
                model.reset_parameters()
                results_dict = dict()

                # Fine-tune
                model, results_dict = train_loop("finetune", args.finetune_sourceint_list, model, args, device, training_type2split2source2dataset_or_loader, provider.source2task, results_dict, split2writer)

                # ================== [新增代码：保存模型权重] ==================
                if args.filename != '':
                    # 确保保存的文件夹存在
                    os.makedirs(os.path.dirname(args.filename), exist_ok=True)
                    if args.use_5fold_cv:
                        save_path = args.filename.replace('.pt', f'_fold_{fold_idx + 1}_model_run_{run}.pt')
                    else:
                        # 例如将 "results/rule_mask.pt" 变成 "results/rule_mask_model_run_0.pt"
                        save_path = args.filename.replace('.pt', f'_model_run_{run}.pt')
                else:
                    save_path = f"best_model_fold_{fold_idx + 1}_run_{run}.pt" if args.use_5fold_cv else f"best_model_run_{run}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"\n[INFO] 已成功保存当前 run 的纯净模型权重至: {save_path}\n")
                # ==============================================================

                if args.dataset_name == 'CUSTOM':
                    # 定义疾病名称 (必须与 excel2pt.py 一致)
                    DISEASE_NAMES = [
                        '冠状动脉粥样硬化', '冠状动脉粥样硬化性心脏病', '不稳定型心绞痛',
                        'ST段抬高型心肌梗死', '非ST段抬高型心肌梗死', '稳定型心绞痛',
                        '隐匿性或无症状型心肌缺血', '缺血性心肌病'
                    ]
                    # 执行替换
                    rename_jaccard_keys_in_dict(results_dict, DISEASE_NAMES)

                # Add to list of results_dict (i.e. one per run/fold)
                results_dict_list.append(results_dict)
            
        # Save overall
        agg_results_dict = save_results(results_dict_list, training_type2split2source2dataset_or_loader, args, SPLITS)
        torch.save(results_dict_list, args.filename.split(".pt")[0]+"_results_dict_list.pt")

        # 训练结果保存为 result.pt 后，自动生成同目录下的 result.json 和 疾病结果.xlsx。
        # 这里不改变训练结果本身，只做结果格式转换，方便后续直接查看各疾病指标。
        if args.filename != '':
            try:
                convert_and_export(args.filename)
            except Exception as e:
                print(f"[WARN] 自动导出 result.json / 疾病结果.xlsx 失败: {e}")

        # assert(len(args.finetune_sourceint_list) == 1)
        # sourceint = args.finetune_sourceint_list[0]
        # val_acc = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val']['pearsonr'])
        # print(f'Val pearsonr: {val_acc.mean()} pm {val_acc.std()}')
        # test_acc = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test']['pearsonr'])
        # print(f'Test pearsonr: {test_acc.mean()} pm {test_acc.std()}')

        # === 修改开始：根据任务类型打印不同的指标 ===
    
        # 获取当前的数据来源 ID (对于自定义数据通常是 0)
        assert(len(args.finetune_sourceint_list) == 1)
        sourceint = args.finetune_sourceint_list[0]
        
        print("\n" + "="*40)
        print(f"Final Evaluation Results (Dataset: {args.dataset_name})")
        print("="*40)

        # 1. 针对自定义多标签分类任务 (CUSTOM)
        if args.dataset_name == 'CUSTOM':
            # # 定义疾病名称列表
            # DISEASE_NAMES = [
            #     '冠状动脉粥样硬化', 
            #     '冠状动脉粥样硬化性心脏病', 
            #     '不稳定型心绞痛', 
            #     'ST段抬高型心肌梗死', 
            #     '非ST段抬高型心肌梗死', 
            #     '稳定型心绞痛', 
            #     '隐匿性或无症状型心肌缺血', 
            #     '缺血性心肌病'
            # ]
            # # 尝试提取并打印 AUROC 、 Accuracy 、 Jaccard
            # metrics_to_print = ['auroc', 'acc', 'jaccard_samples']
            
            # # 检查结果字典中是否存在这些指标
            # val_metrics = agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val']
            
            # found_metric = False
            # for metric in metrics_to_print:
            #     if metric in val_metrics:
            #         found_metric = True
            #         # 获取多次运行的平均值和标准差
            #         val_scores = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val'][metric])
            #         test_scores = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test'][metric])
                    
            #         print(f"Metric: {metric.upper()}")
            #         print(f"  Val : {val_scores.mean():.4f} ± {val_scores.std():.4f}")
            #         print(f"  Test: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
            #         print("-" * 20)
            
            # # 如果您还想看每类疾病的详细 Jaccard，可以额外加一段循环：
            # print("\n[Per-Class Jaccard Details]")
            # # 假设您有 3 类
            # # for i in range(3): 
            # #     key = f"jaccard_class_{i}"
            # for disease_name in DISEASE_NAMES:
            #     key = disease_name
            #     if key in val_metrics:
            #         val_s = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val'][key])
            #         test_s = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test'][key])
            #         # print(f"  Class {i}: Val {val_s.mean():.4f} | Test {test_s.mean():.4f}")
            #         print(f"  {disease_name:<15}: Val {val_s.mean():.4f} | Test {test_s.mean():.4f}")

            # if not found_metric:
            #     print("Warning: No classification metrics (auroc/acc) found in results.")
            #     print("Available keys:", val_metrics.keys())
            DISEASE_NAMES = [
                '冠状动脉粥样硬化', '冠状动脉粥样硬化性心脏病', '不稳定型心绞痛', 
                'ST段抬高型心肌梗死', '非ST段抬高型心肌梗死', '稳定型心绞痛', 
                '隐匿性或无症状型心肌缺血', '缺血性心肌病'
            ]
            
            # (1) 打印整体指标 (Overall)
            print("\n[Overall Metrics]")
            # 添加了新指标
            metrics_to_print = [
                'auroc_micro', 'auroc_samples', 'auroc_macro',
                'auprc_micro', 'auprc_macro', 'acc',
                'precision_macro', 'recall_macro', 'f1_macro', 'specificity_macro', 'fpr_macro', 'fnr_macro',
                'precision_micro', 'recall_micro', 'f1_micro', 'jaccard_micro',
                'precision_samples', 'recall_samples', 'f1_samples', 'jaccard_samples'
            ]
            
            val_metrics = agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val']
            
            for metric in metrics_to_print:
                if metric in val_metrics:
                    val_s = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val'][metric])
                    test_s = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test'][metric])
                    # 格式化打印
                    print(f"  {metric:<20}: Val {np.nanmean(val_s):.4f} | Test {np.nanmean(test_s):.4f}")
            print("-" * 60)

            # (2) 打印每种疾病的详细指标 (Per-Disease)
            print("\n[Per-Disease Detailed Metrics (Test Set)]")
            # 表头
            header = f"{'Disease Name':<15} | {'AUROC':<6} | {'AUPRC':<6} | {'ACC':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Jaccard':<6} | {'Spec':<6} | {'FPR':<6} | {'FNR':<6}"
            print(header)
            print("-" * len(header))

            # 遍历疾病
            for i, disease in enumerate(DISEASE_NAMES):
                # 构造 key。注意：因为 evaluator.py 产生的是 'auroc_class_0' 这种 key
                # 而上面的 rename 函数可能已经把它改名了。
                # 为了打印时的稳健性，我们直接去 save.pt 结果字典里找被 rename 后的 key。
                # 按照 rename 函数的逻辑，key 是 "{Metric}_{Disease}" (例如 "Auroc_冠状动脉硬化")
                
                # 定义要提取的指标前缀 (对应 rename 函数里的 metric_name)
                # Jaccard, Precision, Recall, F1, Auroc, Acc
                
                # 辅助函数：获取某指标的 Test 平均分
                def get_score(metric_prefix, disease_name):
                    # 尝试查找 Metric_Disease 格式 (首字母大写)
                    metric_name_map = {
                        'jaccard': 'Jaccard',
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1': 'F1',
                        'auroc': 'Auroc',
                        'auprc': 'Auprc',
                        'acc': 'Acc',
                        'specificity': 'Specificity',
                        'fpr': 'FPR',
                        'fnr': 'FNR',
                    }
                    key = f"{metric_name_map.get(metric_prefix, metric_prefix.capitalize())}_{disease_name}"
                    if key in agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test']:
                         scores = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test'][key])
                         return scores.mean()
                    return 0.0

                auroc = get_score("auroc", disease)
                auprc = get_score("auprc", disease)
                acc = get_score("acc", disease)
                prec = get_score("precision", disease)
                rec = get_score("recall", disease)
                f1 = get_score("f1", disease)
                jac = get_score("jaccard", disease)
                spec = get_score("specificity", disease)
                fpr = get_score("fpr", disease)
                fnr = get_score("fnr", disease)
                
                # 打印行
                print(f"{disease:<15} | {auroc:.4f} | {auprc:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {jac:.4f} | {spec:.4f} | {fpr:.4f} | {fnr:.4f}")
            
            print("="*60 + "\n")

            # # === 修改 2: (3) 打印风险分级详细指标 (Risk Stratification) ===
            # print("\n[Risk Stratification Metrics (Test Set)]")
            # print("  * Prev_H: 高风险组真实患病率 (越大越好)")
            # print("  * Lift:   提升度 (High / Overall, 越大越好)")
            # print("  * OR:     优势比 (High vs Low, 越大越好)")
            # print("  * NNS:    需筛查人数 (1 / Prev_H, 越小越好)")
            
            # # 定义表头
            # header_risk = f"{'Disease Name':<15} | {'AUROC':<6} | {'Prev_H':<6} | {'Prev_L':<6} | {'Lift':<6} | {'OR':<6} | {'NNS':<5}"
            # print("-" * len(header_risk))
            # print(header_risk)
            # print("-" * len(header_risk))

            # for i, disease in enumerate(DISEASE_NAMES):
            #     auroc = get_score("Auroc", disease)
            #     prev_h = get_score("Prev_high", disease) 
            #     prev_m = get_score("Prev_mid", disease)
            #     prev_l = get_score("Prev_low", disease)
            #     lift = get_score("Lift", disease)
            #     odds_r = get_score("OR", disease)
            #     nns = get_score("NNS", disease)
                
            #     print(f"{disease:<15} | {auroc:.3f}  | {prev_h:.3f}  | {prev_m:.3f}  | {prev_l:.3f}  | {lift:.2f}   | {odds_r:.2f}   | {nns:.1f}")
            
            # print("="*80 + "\n")
            # # ==========================================================

            # === 修改 3: 打印双层筛查分级表格 (Unified Noisy-OR) ===
            print("\n[Coronary Artery Disease Progression Screening]")
            print("  * Logic:  Unified Noisy-OR (Probabilistic Union)")
            print("  * Levels: Level 1 (Diagnosis >= 50%) vs Level 2 (Abnormality > 0%)")
            print("-" * 110)
            
            header_risk = f"{'Screening Target':<20} | {'AUROC':<8} | {'Prev_H':<8} | {'Lift':<6} | {'OR':<6} | {'NNS':<6} | {'Definition'}"
            print(header_risk)
            print("-" * 110)

            groups = [
                ("ACS",               "急性冠脉综合征 (Sub-Risk)"),
                ("CCS",               "慢性冠脉综合征 (Sub-Risk)"),
                ("CHD_Diagnosis",     "Level 1: 冠心病确诊 (Stenosis >= 50%)"),
                ("Coronary_Abnormal", "Level 2: 冠脉血管异常 (Stenosis > 0%)")
            ]
            
            test_metrics = agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test']

            for key, desc in groups:
                def get_m(metric_name):
                    full_key = f"{metric_name}_{key}" 
                    if full_key in test_metrics:
                        return np.mean(test_metrics[full_key])
                    return 0.0

                print(f"{key:<20} | {get_m('auroc'):.4f}   | {get_m('prev_high'):.4f}   | {get_m('lift'):.2f}   | {get_m('or'):.2f}   | {get_m('nns'):.1f}    | {desc}")
            
            print("-" * 110)
            print("\n")
            # ==========================================================

        # 2. 针对原始回归任务 (BRCA 等)
        else:
            # 保持原有的 PearsonR 打印逻辑，但加一个安全检查
            if 'pearsonr' in agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val']:
                val_acc = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val']['pearsonr'])
                print(f'Val pearsonr: {val_acc.mean()} pm {val_acc.std()}')
                
                test_acc = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test']['pearsonr'])
                print(f'Test pearsonr: {test_acc.mean()} pm {test_acc.std()}')
            else:
                print("Warning: 'pearsonr' not found. Available keys:", 
                    agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val'].keys())

        print("="*40 + "\n")
