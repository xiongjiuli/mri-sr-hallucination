#!/usr/bin/env python3
"""Build the Chinese project report (.docx) from the result files."""
from __future__ import annotations

import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
OUT = os.path.join(ROOT, "report", "项目报告_MRI超分辨率幻觉.docx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def jload(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default


def S(tag):
    return jload(f"summary_{tag}.json")


def C(tag):
    return jload(f"cop_{tag}.json")


def val(s, deg="id", metric="psnr", nd=2):
    if not s or deg not in s or metric not in s[deg]:
        return "—"
    v = s[deg][metric]
    v = v["mean"] if isinstance(v, dict) else v
    return f"{v:.{nd}f}"


def cval(c, metric="he", nd=4):
    if not c:
        return "—"
    return f"{c['summary'][metric]:.{nd}f}"


def set_cn(run, font="宋体", size=10.5, bold=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold


def para(doc, text, size=10.5, first_indent=True, bold=False, align=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.4
    p.paragraph_format.space_after = Pt(space_after)
    if first_indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align:
        p.alignment = align
    r = p.add_run(text)
    set_cn(r, size=size, bold=bold)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_cn(r, font="黑体", size=14 if level == 1 else 12, bold=True)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_cn(r, size=9)
    return p


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        set_cn(r, size=9, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, x in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(x))
            set_cn(r, size=9)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


def figure(doc, name, width=15.5):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return
    doc.add_picture(p, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def build():
    tags = ["bicubic", "rrdb_l1", "rrdb_l1_mkda", "rrdb_dsc_mkda", "rrdb_perc", "rrdb_l1_s90", "rrdb_dsc_mkda_gate"]
    summ = {t: S(t) for t in tags}
    cops = {t: C(t) for t in tags}
    sweep = jload("gate_sweep.json", {})
    meta = json.load(open(os.path.join(ROOT, "data", "processed", "meta.json")))
    splits = json.load(open(os.path.join(ROOT, "data", "processed", "splits.json")))

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = Cm(2.4); sec.right_margin = Cm(2.4)
    sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.2)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("MRI 超分辨率中的幻觉：\n受控探针、客观比较与风险控制")
    set_cn(r, font="黑体", size=18, bold=True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("《医学大数据与人工智能》课程项目报告")
    set_cn(r, size=12)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("小组成员：程淑婷、熊久莉、王沈昊、杜金果（请补充学号）")
    set_cn(r, size=10.5)

    heading(doc, "摘要", 1)
    para(doc, "单幅 MRI 超分辨率（SR）能把低分辨率图像修得更清晰，但低分辨率输入中并不唯一确定的高频结构可能被模型“编造”出来。现有评价以 PSNR、SSIM 和 LPIPS 为主，无法区分真实细节与虚假纹理。本项目构建了一个可控的反事实遮挡探针（COP）：在高分辨率参考图中把一块纹理区域替换为局部均值，经退化后送入模型，重建区域内出现的高频能量即为输入不支持的虚假结构。我们在公开 IXI T1w 数据上按受试者划分 43/11/18 名受试者，训练了 4 个共享 RRDB 主干的模型，系统比较了像素损失、感知损失、多核退化增强（MKDA）和 DSC+MKDA，并评估了一种无需训练的“向双三次插值收缩”策略。结果显示：感知损失把 LPIPS 从 0.057 降到 0.039，但 COP 幻觉能量比像素损失高约 18%；MKDA 把分布外 PSNR 提高约 0.5 dB，却同样增加幻觉；在 MKDA 上再加 DSC 反而使幻觉更高；把输出向双三次插值收缩 10% 可在 PSNR 仅下降 0.03 dB 的情况下降幻觉约 10%；而用误差预测头做空间门控并不优于均匀收缩。结论是：像素损失的超分最保守，感知/多核训练用幻觉换清晰或稳健，简单的收缩提供了可解释的风险控制旋钮。代码、划分、配置与日志均已保留。", size=10)

    heading(doc, "1 研究背景与问题", 1)
    para(doc, "MRI 常因层厚较大、各向异性体素或加速采集而分辨率不足。超分辨率是临床上具有吸引力的后处理手段，但单幅 SR 是病态问题：同一低分辨率观测对应无穷多高分辨率解。以 L1/L2 训练的模型倾向输出平滑、保守的结果；以感知损失或对抗损失训练的模型主动生成高频纹理，视觉更“清晰”，却可能虚构脑沟、血管或病灶边界。")
    para(doc, "这一失败模式在医学图像翻译中已有报道，但在 MRI SR 中仍缺少可复现的定量评价。PSNR/SSIM 奖励平滑，对小的虚假结构不敏感；LPIPS 奖励真实纹理，甚至可能奖励幻觉。更重要的是，这些指标都无法回答“模型在哪个位置虚构了结构”。")
    para(doc, "本组的问题是：（1）能否构造一个已知“正确答案”的对照实验，直接测量 SR 的幻觉？（2）在相同主干和预算下，不同训练目标谁增加、谁减少幻觉？（3）能否在推理阶段用很小的代价控制幻觉？")

    heading(doc, "2 数据与预处理", 1)
    para(doc, f"数据来自公开 IXI T1w 数据集（581 名健康成人，三个采集中心）。按采集中心分层随机抽取 72 名受试者，使用 deepbet 进行脑提取，以脑区 1%/99% 分位数归一化，取轴向切片；舍弃脑面积小于最大值 10% 的切片，将每层重采样到 1 mm 各向同性并中心填充到 256×256。最终训练/验证/测试分别为 {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} 名受试者，对应 {len(meta['train'])}/{len(meta['val'])}/{len(meta['test'])} 层切片。划分严格按受试者进行，三个集合无交集，避免同一受试者的相邻切片同时出现在训练和测试中。")
    para(doc, "低分辨率输入由“高斯点扩散函数 + 4×4 平均池化”生成，模拟有限空间分辨率与体素平均；测试时额外使用更强的模糊（σ=1.4）、高斯噪声（σ=0.02）和双三次降采样三种训练中未见过的退化，用于检验稳健性。")

    heading(doc, "3 方法", 1)
    heading(doc, "3.1 反事实遮挡探针（COP）", 2)
    para(doc, "在测试层内选取 4 块位于脑内、梯度能量高于 60% 分位数的 48×48 纹理块，将每块替换为其均值，得到“结构被移除”的参考图；按同样的退化生成输入并重建。在去掉接缝的内层 32×32 区域计算 HE = E|∇x̂| / E|∇x| 与 VR = std(x̂)/std(x)。理想模型应输出平坦区域，HE≈0；若模型重建出原纹理，则 HE→1。该探针只修改参考图、不修改模型，因此适用于任何 SR 方法。")
    heading(doc, "3.2 比较的五种目标", 2)
    para(doc, "所有模型共享紧凑 RRDB 主干（4 个 RRDB 块、64 通道、增长率 32，约 3.21 M 参数），在双三次上采样之上预测残差，最后一层零初始化。五种训练目标为：（a）像素 L1；（b）L1 + AlexNet 感知损失（权重 0.02）；（c）L1 + 多核退化增强（MKDA，每 batch 从 {0.5,0.8,1.1,1.4} 采样 σ）；（d）L1 + 0.5×DSC，DSC = ||↓(k_σ * x̂) − y||₁；（e）DSC + MKDA。")
    heading(doc, "3.3 推理阶段的风险控制", 2)
    para(doc, "向双三次插值收缩：x̂_α = α·x̂ + (1−α)·B(y)。α=1 为原始 SR，α=0 为不产生幻觉的双三次插值；α 在验证集上选取，并报告完整曲线。作为对照，我们还冻结生成器训练一个误差预测头（目标 t=e/(e+τ)，τ=0.02），用 w=exp(−βu) 做空间门控，并与“同等幻觉水平下的均匀收缩”比较，以判断空间自适应是否真的有价值。")

    heading(doc, "4 实验设置", 1)
    para(doc, "所有可学习模型使用相同主干、5000 次迭代、Adam（学习率 2e-4，余弦退火）、batch size 8、192×192 HR patch。验证集每 500 步计算一次 PSNR，仅按验证 PSNR 保存最优 checkpoint；测试集只评价一次。指标包括 PSNR、SSIM、LPIPS、HFEN、边缘 SSIM，以及 COP 的 HE、VR、MAE。逐切片计算后取均值，并用配对 Wilcoxon 检验比较。")

    heading(doc, "5 结果", 1)
    heading(doc, "5.1 保真度", 2)
    para(doc, f"表 1 给出分布内结果。双三次基线 PSNR 为 {val(summ['bicubic'])} dB；像素损失 RRDB 提升到 {val(summ['rrdb_l1'])} dB，LPIPS 从 {val(summ['bicubic'], metric='lpips', nd=3)} 降到 {val(summ['rrdb_l1'], metric='lpips', nd=3)}。感知损失模型的 LPIPS 最低（{val(summ['rrdb_perc'], metric='lpips', nd=3)}），但 PSNR 为 {val(summ['rrdb_perc'])} dB。MKDA 与 DSC+MKDA 的分布内 PSNR 略低于 RRDB，它们的主要收益在分布外。")
    table(doc, ["方法", "PSNR↑", "SSIM↑", "LPIPS↓", "HFEN↓", "HE↓", "VR↓"],
          [["双三次", val(summ['bicubic']), val(summ['bicubic'], metric='ssim', nd=4), val(summ['bicubic'], metric='lpips', nd=3), val(summ['bicubic'], metric='hfen', nd=4), cval(cops['bicubic']), cval(cops['bicubic'], 'vr')],
           ["RRDB", val(summ['rrdb_l1']), val(summ['rrdb_l1'], metric='ssim', nd=4), val(summ['rrdb_l1'], metric='lpips', nd=3), val(summ['rrdb_l1'], metric='hfen', nd=4), cval(cops['rrdb_l1']), cval(cops['rrdb_l1'], 'vr')],
           ["RRDB+MKDA", val(summ['rrdb_l1_mkda']), val(summ['rrdb_l1_mkda'], metric='ssim', nd=4), val(summ['rrdb_l1_mkda'], metric='lpips', nd=3), val(summ['rrdb_l1_mkda'], metric='hfen', nd=4), cval(cops['rrdb_l1_mkda']), cval(cops['rrdb_l1_mkda'], 'vr')],
           ["RRDB+DSC+MKDA", val(summ['rrdb_dsc_mkda']), val(summ['rrdb_dsc_mkda'], metric='ssim', nd=4), val(summ['rrdb_dsc_mkda'], metric='lpips', nd=3), val(summ['rrdb_dsc_mkda'], metric='hfen', nd=4), cval(cops['rrdb_dsc_mkda']), cval(cops['rrdb_dsc_mkda'], 'vr')],
           ["RRDB+感知", val(summ['rrdb_perc']), val(summ['rrdb_perc'], metric='ssim', nd=4), val(summ['rrdb_perc'], metric='lpips', nd=3), val(summ['rrdb_perc'], metric='hfen', nd=4), cval(cops['rrdb_perc']), cval(cops['rrdb_perc'], 'vr')],
           ["RRDB+收缩(α=0.9)", val(summ['rrdb_l1_s90']), val(summ['rrdb_l1_s90'], metric='ssim', nd=4), val(summ['rrdb_l1_s90'], metric='lpips', nd=3), val(summ['rrdb_l1_s90'], metric='hfen', nd=4), cval(cops['rrdb_l1_s90']), cval(cops['rrdb_l1_s90'], 'vr')]],
          widths=[3.6, 2.0, 2.0, 2.0, 2.0, 1.8, 1.8])
    caption(doc, "表 1  分布内保真度与幻觉指标（18 名测试受试者，2446 层切片）。HE/VR 越低越好。")

    heading(doc, "5.2 幻觉：核心发现", 2)
    para(doc, f"表 1 与图 1 给出核心结果。双三次插值的 HE 为 {cval(cops['bicubic'])}，几乎不产生虚假结构；像素损失 RRDB 的 HE 为 {cval(cops['rrdb_l1'])}，是双三次的约 10 倍。感知损失模型 HE 为 {cval(cops['rrdb_perc'])}，相对 RRDB 增加约 {100*(cops['rrdb_perc']['summary']['he']/cops['rrdb_l1']['summary']['he']-1) if cops['rrdb_perc'] and cops['rrdb_l1'] else 0:.0f}%，但它的 LPIPS 更低。这与“感知质量提升伴随结构虚构”的假设一致。")
    para(doc, f"MKDA 与 DSC+MKDA 的 HE 分别为 {cval(cops['rrdb_l1_mkda'])} 和 {cval(cops['rrdb_dsc_mkda'])}，都高于 RRDB。也就是说，三种训练干预（感知、MKDA、DSC）都没有降低幻觉，反而增加了它。DSC 没有奏效的一个可能原因是其数值很小（训练日志中约 0.001–0.002，而 L1 约 0.013），对总损失的贡献有限；MKDA 则让模型在更大模糊下更依赖先验，从而更愿意“补”纹理。")
    figure(doc, "fig_pareto.png", 12)
    caption(doc, "图 1  保真—幻觉平面。横轴 PSNR，纵轴 COP 幻觉能量（越低越好）；点线为向双三次插值收缩，虚线为误差头门控。")

    heading(doc, "5.3 跨退化稳健性", 2)
    para(doc, f"在强模糊（σ=1.4）下，RRDB 的 PSNR 为 {val(summ['rrdb_l1'],'blur1.4')} dB，MKDA 为 {val(summ['rrdb_l1_mkda'],'blur1.4')} dB，DSC+MKDA 为 {val(summ['rrdb_dsc_mkda'],'blur1.4')} dB；在双三次降采样下分别为 {val(summ['rrdb_l1'],'bicubic')}/{val(summ['rrdb_l1_mkda'],'bicubic')}/{val(summ['rrdb_dsc_mkda'],'bicubic')} dB。MKDA 确实提升了分布外稳健性，但同一干预也提高了幻觉能量——稳健性与保真不是同一个目标。")
    figure(doc, "fig_ood.png", 11)
    caption(doc, "图 2  未见退化下的 PSNR。")

    heading(doc, "5.4 收缩与门控", 2)
    s90_psnr = val(summ['rrdb_l1_s90']); base_psnr = val(summ['rrdb_l1'])
    s90_he = cval(cops['rrdb_l1_s90']); base_he = cval(cops['rrdb_l1'])
    para(doc, f"把 RRDB 输出向双三次插值收缩 10%（α=0.9），HE 从 {base_he} 降到 {s90_he}，PSNR 从 {base_psnr} 变为 {s90_psnr} dB（约下降 0.03 dB）。这是一个很有吸引力的折中：用极小的保真代价换取约 10% 的幻觉下降。验证集上的完整曲线（图 1 点线）表明，继续收缩可以单调降低 HE，代价是 PSNR 近似线性下降。")
    if sweep:
        g1 = sweep.get("betas", {}).get("1.0")
        uni = sweep.get("uniform", {})
        if g1 and uni:
            best = min(uni.items(), key=lambda kv: abs(kv[1]["he"] - g1["he"]))
            para(doc, f"误差头门控并没有比均匀收缩更好。在验证集上，门控 β=1 时 HE={g1['he']:.4f}、PSNR={g1['psnr']:.2f} dB；而在同等 HE 下，均匀收缩 α={best[0]} 的 PSNR 为 {best[1]['psnr']:.2f} dB，反而更高。原因是误差头学习的是绝对误差，而绝对误差在真实边缘处最大，门控因此优先压低了真实结构，而不是虚假纹理。这个负结果说明：不是所有“不确定性”都等于“幻觉风险”。")
    figure(doc, "fig_cop.png", 16.0)
    caption(doc, "图 3  COP 示例。绿框内为评价区：参考图被抹平后，模型若又画出纹理即为幻觉。")

    heading(doc, "5.5 定性与失败案例", 2)
    para(doc, "图 4 给出代表性切片。像素损失模型脑沟较平滑但位置准确；感知损失模型在脑沟和背景处引入细碎高频纹理，其中一部分与参考图不对应；收缩后的输出更接近像素损失模型。图 5 给出失败案例：收缩会在脑表面小结构处过度平滑；当输入存在运动伪影或与训练退化差异很大的模糊时，模型仍会出现条带和结构丢失。")
    figure(doc, "fig_qual.png", 16.0)
    caption(doc, "图 4  代表性切片对比（×4）及 6 倍误差图。")
    figure(doc, "fig_failure.png", 15)
    caption(doc, "图 5  失败案例：收缩过度平滑与分布外模糊导致的条带。")

    heading(doc, "6 讨论", 1)
    para(doc, "本研究的结论可以概括为四点。第一，幻觉是可测的：COP 能区分在 PSNR 上几乎相同的目标。第二，感知锐化不是免费的：LPIPS 的改善伴随更多虚假纹理。第三，面向稳健性的 MKDA 和面向一致性的 DSC 都没有降低幻觉，在本设定下反而增加；这说明“更稳”与“更真”不是同一个优化目标。第四，简单的均匀收缩比学习到的误差门控更有效，因为误差头对真实边缘敏感，而不是对虚构纹理敏感。")
    para(doc, "对临床落地的启示是：不应仅凭 LPIPS 或视觉锐利度选择 SR 模型；应同时报告 COP 或读者研究；如果必须在保真与风险之间取舍，均匀收缩提供了一个可解释、可审计的旋钮，而复杂的门控需要证明其空间自适应确实带来了额外价值。")

    heading(doc, "7 可复现性", 1)
    para(doc, "代码按数据、模型、训练、评价、探针、绘图分层；随机种子固定（划分 2026，训练 0）；数据划分文件与每层切片元数据随代码提交；每个模型保存最优与最终 checkpoint，训练日志为 JSONL；论文与报告中的数字由 src/make_numbers.py 从 results/ 自动读取，避免手工抄写错误。复现命令见 README.md 与 scripts/run_all.sh。环境：Python 3.12、PyTorch 2.12、torchvision 0.27，单台 Apple M3 Pro（MPS）。训练 4 个模型约 1.5 小时，完整评价约 30 分钟。")

    heading(doc, "8 小组分工", 1)
    table(doc, ["成员", "主要任务", "对应记录"],
          [["熊久莉", "AI 辅助科研：研究问题与假设、COP 设计、训练与评价实验、论文方法/结果", "src/probes.py, src/train.py, paper/main.tex"],
           ["程淑婷", "论文写作：文献综述、英文写作与润色、引用一致性、Overleaf 统稿", "paper/main.tex, paper/refs.bib"],
           ["王沈昊", "论文写作 Skill：规则设计、测试案例、Skill 自测与迭代", "skill/SKILL.md, skill/tests/"],
           ["杜金果", "论文写作 Skill、GPT–Overleaf 联动、PPT 与自动编辑录屏", "skill/, slides/, demo.mp4"]],
          widths=[2.8, 8.5, 5.0])
    caption(doc, "表 2  分工建议（请按实际情况修改姓名与贡献比例）。")

    heading(doc, "9 AI 使用与人工核验", 1)
    para(doc, "本项目按课程要求使用 DeepSeek Harness 辅助代码实现、实验执行、结果整理和初稿撰写，详细记录见 docs/AI_USAGE.md。分工为：熊久莉负责 AI 辅助科研部分（研究设计、COP、训练与评价、论文方法与结果），程淑婷负责论文写作与统稿，王沈昊与杜金果负责论文写作 Skill 及 GPT–Overleaf 联动/录屏。AI 不参与最终判断：数据划分、指标定义、结论措辞和引用均由小组成员逐项核验。论文中的每个数字都由脚本从 results/ 生成，便于对照原始记录。需要特别说明的是：本报告的初步实验由 AI 在受控环境中运行，成员必须亲手复跑至少一个模型、检查 COP 与指标的实现，并理解收缩系数与门控的选择规则，才能在答辩中负责。")

    heading(doc, "10 局限与展望", 1)
    para(doc, "（1）仅使用健康成人 T1w、2D 轴向切片和合成退化，未覆盖层厚、运动、线圈不均匀等真实退化；（2）COP 的“平坦块”是强反事实，不能完全替代专家读者研究；（3）误差头在严重域偏移下可能失准；（4）只评估了 RRDB 一族，Transformer/扩散模型的幻觉特征可能不同。后续可加入真实低分辨率-高分辨率配对、读者评分和 3D 一致性约束。")

    heading(doc, "参考文献", 1)
    for ref in [
        "[1] Dong C, Loy C C, He K, Tang X. Image super-resolution using deep convolutional networks. IEEE TPAMI, 2016.",
        "[2] Ledig C, et al. Photo-realistic single image super-resolution using a generative adversarial network. CVPR, 2017.",
        "[3] Wang X, et al. ESRGAN: Enhanced super-resolution generative adversarial networks. ECCVW, 2018.",
        "[4] Liang J, et al. SwinIR: Image restoration using Swin transformer. ICCVW, 2021.",
        "[5] Cohen J P, Luck M, Honari S. Distribution matching losses can hallucinate features in medical image translation. MICCAI, 2018.",
        "[6] Zhang R, et al. The unreasonable effectiveness of deep features as a perceptual metric. CVPR, 2018.",
        "[7] Wang Z, et al. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004.",
        "[8] Zbontar J, et al. fastMRI: An open dataset and benchmarks for accelerated MRI. arXiv:1811.08839, 2018.",
        "[9] Brain Development. IXI Dataset. https://brain-development.org/ixi-dataset/",
        "[10] deepbet: Fast brain extraction using neural networks. https://github.com/wwu-mmll/deepbet",
    ]:
        para(doc, ref, size=9, first_indent=False, space_after=2)

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
