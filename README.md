## CustomData

#### baseline

```
python single_source_pipeline.py --filename results/baseline.pt \
  --device 0 --epochs 30 --runs 3 --model GGMLP \
  --l1_weight 0 --l2_weight 0 --batch_size 32 \
  --simple --seed 0 --drug_div 1 --gene_div 1000 \
  --mlp_m_layer_list 32,32,8 \
  --enlarge 1 --lr 0.001 --mp --beta 0.1 \
  --load_dir ../data/kg --cache_dir ../data --dataset CUSTOM
```

<!-- 类别权重 -->
--use_class_weights

<!-- 双向自适应采样 -->
--use_bidirectional_adaptive_sampling

<!-- 5折交叉验证 -->
--use_5fold_cv

<!-- 选模指标 -->
--selection_metric balanced_macro_score

<!-- 协同不平衡学习 -->
--use_class_weights --use_disease_typed_bidirectional_sampling

<!-- 决策校准 -->
<!-- 疾病级阈值优化 -->
--use_disease_threshold_optimization