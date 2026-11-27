# PKR



## 目录

```text
PKR_clean/
├── main.py                 # 入口
├── trainer.py              # 任务循环、日志、平均准确率与遗忘率
├── exps/                   # 实验配置
├── models/
│   ├── base.py             # 增量学习基类（herding 样本选择、CNN/NME 评估）
│   └── pkr.py              # PKR 方法
├── utils/
│   ├── inc_net.py          # DERNet（逐任务扩展的特征提取器 + 辅助分类器 + 权重对齐）
│   ├── data.py             # CIFAR-10 / CIFAR-100 / TinyImageNet-200 / PathMNIST
│   ├── data_manager.py
│   ├── factory.py
│   └── toolkit.py
└── convs/
    ├── cifar_resnet.py     # ResNet-32（与原文件完全一致）
    ├── resnet.py           # ResNet-18（仅保留用到的部分）
    └── linears.py          # SimpleLinear
```

初始任务 200 epoch（lr 0.1，milestones [60,120,170]，wd 5e-4），
后续任务 170 epoch（lr 0.1，milestones [80,120,150]，wd 2e-4），batch size 128，SGD momentum 0.9。

## 运行

```bash
cd PKR_clean
python main.py --config ./exps/pkr_cifar100_b10_inc10.json
```

| 配置 | 设置 |
|---|---|
| `pkr_cifar100_b10_inc10.json` | CIFAR-100，10 个任务，ResNet-32 |
| `pkr_cifar100_b20_inc20.json` | CIFAR-100，5 个任务 |
| `pkr_cifar100_b5_inc5.json` | CIFAR-100，20 个任务 |
| `pkr_cifar10_b2_inc2.json` | CIFAR-10，5 个任务 |
| `pkr_tinyimagenet_b20_inc20.json` | TinyImageNet-200，10 个任务，ResNet-18，seed 2024 |

- CIFAR 数据默认放在 `./data`（不存在会自动下载）；可在配置中加 `"data_root"` 改路径。
- TinyImageNet 需在配置里把 `data_root` 指向 `tiny-imagenet-200` 目录（内含 `train/`、`val/`）。
- PathMNIST 需要另装 `medmnist`；其余数据集不需要。
- `"device": ["-1"]` 表示用 CPU；多个 GPU 写成 `["0","1"]`。
- `seed` 只决定类别顺序；torch 随机种子与原代码一样固定为 1。
- 日志写在 `logs/pkr/<dataset>/<init_cls>/<increment>/`。

依赖：`torch`、`torchvision`、`numpy`、`scipy`、`tqdm`、`Pillow`。
