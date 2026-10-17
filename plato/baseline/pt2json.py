import torch
import json
import numpy as np
import argparse
import math
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def convert_to_serializable(obj):
    """
    辅助函数：将 PyTorch Tensor 和 NumPy 数组转换为 JSON 可识别的列表格式。
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()  # 移至 CPU 并转为列表
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # NumPy 转为列表
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (set, tuple)):
        return list(obj)  # 集合/元组转为列表
    elif hasattr(obj, '__dict__'):
        return obj.__dict__  # 如果是自定义对象，尝试返回其属性字典
    else:
        # 如果遇到无法识别的类型，将其转换为字符串以防报错
        return str(obj)


DISEASE_NAME_GROUPS = [
    ("冠状动脉粥样硬化", ["冠状动脉粥样硬化"]),
    ("冠状动脉粥样硬化性心脏病", ["冠状动脉粥样硬化性心脏病"]),
    ("不稳定型心绞痛", ["不稳定型心绞痛"]),
    ("ST段抬高型心肌梗死", ["ST段抬高型心肌梗死"]),
    ("非ST段抬高型心肌梗死", ["非ST段抬高型心肌梗死"]),
    ("稳定型心绞痛", ["稳定型心绞痛"]),
    ("隐匿性或无症状型心肌缺血", ["隐匿性或无症状型心肌缺血"]),
    ("缺血性心肌病", ["缺血性心肌病"]),
]

DISEASE_METRICS = [
    ("AUROC", "Auroc"),
    ("AUPRC", "Auprc"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1", "F1"),
    ("Specificity", "Specificity"),
    ("FPR", "FPR"),
    ("FNR", "FNR"),
]

MACRO_METRICS = [
    ("Macro-AUROC", "Auroc"),
    ("Macro-AUPRC", "Auprc"),
    ("Macro-Precision", "Precision"),
    ("Macro-Recall", "Recall"),
    ("Macro-F1", "F1"),
    ("Macro-Specificity", "Specificity"),
    ("Macro-FPR", "FPR"),
    ("Macro-FNR", "FNR"),
]


def convert_result_pt_to_json(input_file='results/result.pt', output_file=None):
    """将训练输出的 result.pt 转换为同目录下的 result.json。"""
    input_file = Path(input_file)
    if output_file is None:
        output_file = input_file.with_suffix('.json')
    output_file = Path(output_file)
    os.makedirs(output_file.parent, exist_ok=True)

    print(f"正在加载 {input_file} ...")

    # map_location='cpu' 确保即使没有 GPU 也能加载
    data = torch.load(input_file, map_location='cpu', weights_only=False)
    add_missing_macro_metrics(data)

    print("加载成功，正在转换为 JSON 格式...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用 default 参数处理 Tensor 等非标准类型
        json.dump(data, f, indent=4, default=convert_to_serializable, ensure_ascii=False)

    print(f"转换完成！文件已保存为: {output_file}")
    return output_file


def _column_name(col_idx):
    """将列号转换为 Excel 列名，例如 1 -> A。"""
    name = ""
    while col_idx:
        col_idx, rem = divmod(col_idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _format_excel_value(value):
    """统一处理 Excel 单元格值，避免 nan 变成不可读内容。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if math.isnan(float(value)):
            return "NaN"
    except (TypeError, ValueError):
        pass
    return value


def _write_simple_xlsx(output_file, rows):
    """用标准库生成 xlsx，避免额外依赖 openpyxl 或 xlsxwriter。"""
    output_file = Path(output_file)
    os.makedirs(output_file.parent, exist_ok=True)

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, raw_value in enumerate(row, start=1):
            value = _format_excel_value(raw_value)
            cell_ref = f"{_column_name(col_idx)}{row_idx}"
            style_id = 1 if row_idx == 1 else 0
            if value == "":
                cells.append(f'<c r="{cell_ref}" s="{style_id}"/>')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}" s="{style_id}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{cell_ref}" s="{style_id}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    dimension = f"A1:{_column_name(max(len(row) for row in rows))}{len(rows)}"
    worksheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <dimension ref="{dimension}"/>
 <cols>
  <col min="1" max="1" width="28" customWidth="1"/>
  <col min="2" max="2" width="16" customWidth="1"/>
  <col min="3" max="5" width="14" customWidth="1"/>
  <col min="6" max="7" width="30" customWidth="1"/>
 </cols>
 <sheetData>
  {"".join(sheet_rows)}
 </sheetData>
</worksheet>'''

    workbook_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets>
  <sheet name="疾病结果" sheetId="1" r:id="rId1"/>
 </sheets>
</workbook>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <fonts count="2">
  <font><sz val="11"/><name val="Calibri"/></font>
  <font><b/><sz val="11"/><name val="Calibri"/></font>
 </fonts>
 <fills count="2">
  <fill><patternFill patternType="none"/></fill>
  <fill><patternFill patternType="gray125"/></fill>
 </fills>
 <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
 <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
 <cellXfs count="2">
  <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
  <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
 </cellXfs>
 <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
 <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
 <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

    root_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    workbook_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


def _get_first_source_metric_dict(result_json):
    """读取 agg_source2split2metric2score_mean 中第一个 source 的指标。"""
    finetune = result_json.get("finetune", {})
    source_metrics = finetune.get("agg_source2split2metric2score_mean", {})
    if not source_metrics:
        raise KeyError("result.json 中缺少 finetune/agg_source2split2metric2score_mean")
    source_key = next(iter(source_metrics.keys()))
    return source_metrics[source_key]


def _find_metric_value(split_metrics, metric_prefix, disease_aliases):
    """按疾病别名查找指标，兼容“心脏”和“心脏病”等命名差异。"""
    for disease_name in disease_aliases:
        key = f"{metric_prefix}_{disease_name}"
        if key in split_metrics:
            return split_metrics[key]
    return ""


def _compute_macro_metric_value(split_metrics, metric_prefix):
    """根据 8 个疾病的同一指标计算宏平均，忽略不可定义的 NaN。"""
    values = []
    for _, aliases in DISEASE_NAME_GROUPS:
        value = _find_metric_value(split_metrics, metric_prefix, aliases)
        if value == "":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            values.append(value)

    if not values:
        return "NaN"
    return sum(values) / len(values)


def add_missing_macro_metrics(result_data):
    """兼容旧 result.pt：若缺少 recall_macro / f1_macro，则由 8 个疾病指标补算。"""
    finetune = result_data.get("finetune", {})
    source_metrics = finetune.get("agg_source2split2metric2score_mean", {})
    for _, split2metrics in source_metrics.items():
        for _, split_metrics in split2metrics.items():
            if not isinstance(split_metrics, dict):
                continue
            if "recall_macro" not in split_metrics:
                split_metrics["recall_macro"] = _compute_macro_metric_value(split_metrics, "Recall")
            if "f1_macro" not in split_metrics:
                split_metrics["f1_macro"] = _compute_macro_metric_value(split_metrics, "F1")


def build_disease_result_rows(result_json):
    """构建“疾病结果.xlsx”的表格内容。"""
    source_metric_dict = _get_first_source_metric_dict(result_json)
    rows = [["疾病", "指标", "train", "val", "test", "结果分析", "问题分型"]]

    for display_name, aliases in DISEASE_NAME_GROUPS:
        for metric_idx, (metric_name, metric_prefix) in enumerate(DISEASE_METRICS):
            row = [display_name if metric_idx == 0 else "", metric_name]
            for split in ["train", "val", "test"]:
                split_metrics = source_metric_dict.get(split, {})
                row.append(_find_metric_value(split_metrics, metric_prefix, aliases))
            row.extend(["", ""])
            rows.append(row)

    for metric_name, metric_prefix in MACRO_METRICS:
        row = ["", metric_name]
        for split in ["train", "val", "test"]:
            split_metrics = source_metric_dict.get(split, {})
            row.append(_compute_macro_metric_value(split_metrics, metric_prefix))
        row.extend(["", ""])
        rows.append(row)

    return rows


def export_disease_excel_from_json(json_file='results/result.json', output_file=None):
    """从 result.json 中导出 8 种疾病的 train/val/test 指标 Excel。"""
    json_file = Path(json_file)
    if output_file is None:
        output_file = json_file.parent / "疾病结果.xlsx"
    output_file = Path(output_file)

    with open(json_file, "r", encoding="utf-8") as f:
        result_json = json.load(f)

    rows = build_disease_result_rows(result_json)
    _write_simple_xlsx(output_file, rows)
    print(f"疾病结果 Excel 已保存为: {output_file}")
    return output_file


def convert_and_export(input_file='results/result.pt', output_json=None, output_excel=None):
    """完整执行 result.pt -> result.json -> 疾病结果.xlsx。"""
    json_file = convert_result_pt_to_json(input_file, output_json)
    excel_file = export_disease_excel_from_json(json_file, output_excel)
    return json_file, excel_file


def main():
    parser = argparse.ArgumentParser(description="将 result.pt 转为 result.json，并导出疾病结果 Excel")
    parser.add_argument("--input", default="results/result.pt", help="输入的 result.pt 路径")
    parser.add_argument("--output_json", default=None, help="输出的 result.json 路径，默认与 input 同目录同名")
    parser.add_argument("--output_excel", default=None, help="输出的 疾病结果.xlsx 路径，默认与 result.json 同目录")
    parser.add_argument("--json_only", action="store_true", help="只导出 JSON，不导出疾病 Excel")
    args = parser.parse_args()

    try:
        json_file = convert_result_pt_to_json(args.input, args.output_json)
        if not args.json_only:
            export_disease_excel_from_json(json_file, args.output_excel)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.input}，请确认 result.pt 路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
