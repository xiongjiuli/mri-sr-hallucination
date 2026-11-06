# results/ 目录说明

所有文件由 `src/` 下的脚本生成，**不要手工修改**；论文和报告中的数字都由这些文件派生。

| 文件 | 内容 |
|---|---|
| `summary_<tag>.json` | 每个模型在各退化条件下的指标均值/标准差（PSNR、SSIM、LPIPS、HFEN、edge-SSIM、GL） |
| `metrics_<tag>.csv` | 逐切片指标，用于配对统计与失败案例分析 |
| `cop_<tag>.json` | COP 探针的逐 patch 结果与汇总（HE、VR、MAE） |
| `cop_examples_<tag>.npz` | 用于绘图的 COP 示例（HR、遮挡图、LR、SR、不确定度） |
| `preds/<tag>_<deg>_<idx>.npz` | 指定切片的 SR/HR/LR/不确定度预测 |
| `gate_sweep.json` | 验证集上 β 扫描（PSNR、SSIM、HE、VR）与选中的 β* |
| `corr.json` | 模型级 PSNR/LPIPS 与 HE 的 Spearman 相关 |
| `figures/` | 论文图（PDF 矢量图 + PNG 300 dpi） |
| `inspect_masks.png` | 预处理脑提取抽查图 |
| `test_slices.png` | 预处理后的测试切片抽查图 |

## 复算顺序

```bash
python3 -m src.evaluate ...     # 生成 summary/metrics/preds
python3 -m src.probes ...       # 生成 cop
python3 -m src.gate_sweep ...   # 生成 gate_sweep
python3 -m src.make_numbers     # 生成 paper/numbers.tex + corr.json
python3 -m src.figures          # 生成 figures/
python3 -m src.build_report     # 生成 report/*.docx
```

## 读图提示

- `fig_pareto`：左上角更好（PSNR 高、HE 低）。门控虚线展示“用 PSNR 换 HE”的连续工作点。
- `fig_qual`：每一行是一个测试切片；最后一列是 6 倍误差图，越暗越好。
- `fig_cop`：绿框内是评价区。参考图被抹平后，模型若又画出纹理，即为幻觉。
- `fig_ood`：训练/测试退化不一致时的 PSNR。
- `fig_failure`：PSNR 最差的切片，用来说明方法失效的场景。
