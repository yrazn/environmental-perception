# D-Fire + Indoor Fire Smoke 联合训练结果

## 模型与训练配置

- 模型：YOLO11n
- 初始化权重：D-Fire 清洗数据集训练得到的 `best.pt`
- 训练数据：D-Fire train + Indoor Fire Smoke train
- 联合验证数据：D-Fire val + Indoor Fire Smoke val
- 优化器：SGD
- 初始学习率：0.0005
- 输入尺寸：640
- Batch size：32
- 训练轮数：60
- 最佳轮次：第48轮（按 Ultralytics fitness）
- 最佳权重：`weights/best.pt`

## 联合验证集最终复验

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.677 | 0.593 | 0.629 | 0.323 |
| Smoke | 0.712 | 0.668 | 0.692 | 0.384 |
| Fire | 0.643 | 0.517 | 0.566 | 0.261 |

联合验证集同时包含两个域且 D-Fire 占比较高，不能直接与单一数据集验证指标横向比较；
是否成功应以两个独立测试集的结果为准。

## 独立测试结果

| 模型与测试集 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| D-Fire基线 → D-Fire test | 0.767 | 0.691 | 0.759 | 0.441 |
| Indoor顺序微调 → Indoor test | 0.847 | 0.769 | 0.845 | 0.498 |
| Indoor顺序微调 → D-Fire test | 0.332 | 0.273 | 0.208 | 0.084 |
| **联合模型 → D-Fire test** | **0.766** | **0.686** | **0.750** | **0.433** |
| **联合模型 → Indoor test** | **0.793** | **0.702** | **0.789** | **0.441** |

联合模型的分类结果：

| 测试集 | 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| D-Fire | Smoke | 0.805 | 0.755 | 0.802 | 0.499 |
| D-Fire | Fire | 0.726 | 0.617 | 0.698 | 0.368 |
| Indoor | Smoke | 0.799 | 0.664 | 0.766 | 0.439 |
| Indoor | Fire | 0.788 | 0.740 | 0.812 | 0.443 |

## 结论

1. 仅使用 Indoor 数据顺序微调造成严重灾难性遗忘，D-Fire mAP50 从 0.759 降至 0.208。
2. 联合训练后 D-Fire mAP50 为 0.750，仅比基线下降 0.009，原始能力基本保留。
3. 联合模型在 Indoor test 上达到 mAP50=0.789，获得了有效的室内场景检测能力。
4. 联合训练同时满足 D-Fire mAP50≥0.72、Indoor mAP50≥0.78 的预设目标。
5. 该模型可作为后续隧道域标注、微调和系统接入的当前 RGB 火焰烟雾基准模型。

## 结果目录

- D-Fire 独立测试：`../yolo11n_joint_sgd_on_dfire_test/`
- Indoor 独立测试：`../yolo11n_joint_sgd_on_indoor_test/`

说明：Indoor 数据集没有纯负样本。隧道灯光、车灯、蒸汽和水雾等困难负样本仍需在
后续隧道域数据或 MS-FSDB 数据中补充验证。
