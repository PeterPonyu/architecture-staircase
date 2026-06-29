# arch_staircase — Direction 017: 谁拥有阶梯（架构解剖 × 按阶计时）

方向文档：`directions/017-arch-staircase-anatomy.md`（银行 A1 8.7 消耗，
2026-06-12 确认性核查 PROCEED）。**复用 degree_staircase 全基建**（004 闭环
只读基建，不得修改）；自带训练循环仅为冻结臂介入点 + trainable-only
optimizer——日志字段与 `train_staircase.run` 逐字段镜像（SI 可比性纪律）。

## 模块

| 文件 | 职责 |
|---|---|
| `train_arch.py` | ArchConfig（+freeze 轴）、freeze_component（attn=qkv/proj, mlp=fc1/fc2）、trainable-only AdamW、镜像训练循环、smoke |
| `run_arch_sweep.py` | 网格：宽度（d64=8 seeds 否决者纪律）/深度/冻结臂 + `--heads` P3 子臂；resume + shard |

## 用法

```bash
python run_arch_sweep.py --smoke     # 冻结仪器自检（CPU，零写入）
python run_arch_sweep.py --dry-run   # 打印网格（43 cells；--heads +6）
python run_arch_sweep.py             # 正式网格（归执行者；写 results/arch_staircase/）
```

## 纪律备忘

- **AdamW 固定**——优化器轴属 004（互锁条款），train_arch 内有断言。
- 冻结臂**只报组内 SI 比值与组内 half-time 序**（无参数量配平时禁止跨臂
  绝对值比较——否决者修正原文）。
- 冻结臂若连低阶都不学（trainability 崩溃）→ 该臂不得用于所有权判决
  （kill 判据 2）。
- smoke 自检内容：字段 schema 与 004 trainer 镜像核对、冻结参数训练后
  bit-identical、活组件确实在动、optimizer 参数数 == trainable 数。
