"""Render readable offline HTML and PDF with equations, without network services."""
from pathlib import Path
from io import BytesIO
import base64
import html
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
import markdown
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
RUNTIME_PACKAGES = Path("C:/Users/AQing/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages")
sys.path.append(str(RUNTIME_PACKAGES))
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               Table, TableStyle, XPreformatted, KeepTogether)

EQS = HERE / "equations"
EQS.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont("CJK", "C:/Windows/Fonts/simhei.ttf"))
styles = getSampleStyleSheet()
for name in ("Normal", "BodyText"):
    styles[name].fontName = "CJK"
    styles[name].fontSize = 9.5
    styles[name].leading = 16
    styles[name].wordWrap = "CJK"
    styles[name].spaceAfter = 7
for i in range(1, 4):
    styles[f"Heading{i}"].fontName = "CJK"
    styles[f"Heading{i}"].fontSize = (18, 13, 11)[i-1]
    styles[f"Heading{i}"].leading = (25, 21, 18)[i-1]
    styles[f"Heading{i}"].textColor = colors.HexColor("#173a43")
    styles[f"Heading{i}"].spaceBefore = 12
    styles[f"Heading{i}"].spaceAfter = 8
    styles[f"Heading{i}"].keepWithNext = True
styles.add(ParagraphStyle("Cell", fontName="CJK", fontSize=8, leading=12, wordWrap="CJK"))
styles.add(ParagraphStyle("CodeCJK", fontName="CJK", fontSize=7.3, leading=11, wordWrap="CJK"))
WIDTH = 483


def normalize_math(expr):
    expr = " ".join(expr.split())
    if r"\underbrace" in expr:
        expr = (r"\mathcal{P}_{z_k}V=w_k\alpha_kQ_k+C_k,\quad "
                r"C_k=\sum_{j\ne k}w_j\alpha_j\mathcal{P}_{z_k-z_j}Q_j")
    expr = expr.replace(r"\boxed", "")
    expr = expr.replace(r"\pmod{2\pi}", r"\quad(\mathrm{mod}\ 2\pi)")
    expr = re.sub(r"\\boldsymbol\s*(\\[A-Za-z]+)", lambda m: r"\boldsymbol{"+m[1]+"}", expr)
    for macro in ("mathcal", "mathbf", "widehat", "widetilde"):
        expr = re.sub(r"\\"+macro+r"\s+([A-Za-z])", lambda m: "\\"+macro+"{"+m[1]+"}", expr)
    expr = re.sub(r"\\frac\s*([A-Za-z0-9])", lambda m: r"\frac{"+m[1]+"}", expr)
    expr = re.sub(r"(\\frac\{[A-Za-z0-9]\})\s*([A-Za-z0-9])", r"\1{\2}", expr)
    return expr


def equation(expr, stem, index):
    path = EQS / f"{stem}_{index:02d}.png"
    math_to_image("$"+normalize_math(expr)+"$", path,
                  prop=FontProperties(size=15), dpi=220, format="png", color="#122b34")
    return path


def ptext(text):
    # SimHei does not cover Unicode superscript characters; keep their meaning explicit.
    text = text.replace("⁻²", "^(-2)").replace("⁻¹", "^(-1)").replace("²", "^2")
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#008f86">\1</link>', text)
    text = re.sub(r"\*\*(.*?)\*\*", r'<b>\1</b>', text)
    text = re.sub(r"\x60([^\x60]+)\x60", r"\1", text)
    return text


def pic(path, maxwidth=WIDTH, maxheight=620, native_dpi=None):
    w, h = PILImage.open(path).size
    scale = min(maxwidth/w, maxheight/h)
    if native_dpi is not None:
        scale = min(scale, 72/native_dpi)
    return Image(str(path), width=w*scale, height=h*scale)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("CJK", 8)
    canvas.setFillColor(colors.HexColor("#6c7480"))
    canvas.drawString(56, 25, "Physics proof | 单图仿真；本次无新增训练")
    canvas.drawRightString(539, 25, str(doc.page))
    canvas.restoreState()


def render(name):
    text = (HERE / (name+".md")).read_text(encoding="utf-8")
    # Render every equation once; both output formats use the exact same raster.
    eq_paths = []
    def repl(m):
        path = equation(m[1], name, len(eq_paths)+1)
        eq_paths.append(path)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        width = PILImage.open(path).width*96/220
        return '\n<div class="equation"><img style="width:'+str(width)+'px;max-width:100%" alt="Formula '+str(len(eq_paths))+'" src="data:image/png;base64,'+b64+'"></div>\n'
    html_md = re.sub(r"\$\$(.*?)\$\$", repl, text, flags=re.S)
    body = markdown.markdown(html_md, extensions=["tables", "fenced_code"])
    # Embed result figures, so HTML can be moved and read offline.
    def embed(m):
        src = m[1]
        path = HERE / src
        if not src.startswith("data:") and path.is_file():
            return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode("ascii")+'"'
        return m[0]
    body = re.sub(r'src="([^"]+)"', embed, body)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>"""+name+"""</title>
<style>body{max-width:1080px;margin:48px auto;padding:0 28px;font:16px/1.85 "Microsoft YaHei",sans-serif;color:#243840}
h1,h2,h3{color:#173a43;line-height:1.4}h2{margin-top:42px;border-top:1px solid #dce5e8;padding-top:24px}
a{color:#008f86}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{border-bottom:1px solid #dce5e8;padding:9px;text-align:left}th{background:#edf5f5}
.equation{padding:18px 10px;overflow:auto;text-align:center;background:#f8fafb;margin:18px 0}
.equation img{max-height:115px;width:auto}pre{background:#f3f6f7;padding:16px;white-space:pre-wrap;font-size:13px}
@media print{body{margin:0;max-width:none;font-size:11pt}.equation,table,img{break-inside:avoid}h2{break-after:avoid}}</style>
<body>"""+body+"</body></html>"
    (HERE / (name+".html")).write_text(page, encoding="utf-8")

    lines = text.splitlines()
    story, i, eq_index = [], 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "$$":
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                i += 1
            story.append(pic(eq_paths[eq_index], maxheight=95, native_dpi=220))
            story.append(Spacer(1, 12))
            eq_index += 1
            i += 1
        elif line.startswith("~~~"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("~~~"):
                block.extend(textwrap.wrap(lines[i], width=92, replace_whitespace=False) or [""])
                i += 1
            story.append(KeepTogether([
                XPreformatted(html.escape("\n".join(block)), styles["CodeCJK"]),
                Spacer(1, 10)]))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-:\s]+", c) for c in cells):
                    rows.append([Paragraph(ptext(c), styles["Cell"]) for c in cells])
                i += 1
            n = len(rows[0])
            widths = [WIDTH/n]*n
            if n > 2:
                widths = [WIDTH*.36]+[WIDTH*.64/(n-1)]*(n-1)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7f2f2")),
                ("VALIGN",(0,0),(-1,-1),"TOP"), ("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#c9d8dc")),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            story.append(KeepTogether([table, Spacer(1,12)]))
        elif line.startswith("#"):
            level = min(3, len(line)-len(line.lstrip("#")))
            story.append(Paragraph(ptext(line.lstrip("#").strip()), styles[f"Heading{level}"]))
            i += 1
        elif line.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            story.extend([pic(HERE/match[2], maxheight=590), Spacer(1,10)])
            i += 1
        else:
            para = [line]
            i += 1
            if not line.startswith("- "):
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","~~~","$$","![","- ")):
                    para.append(lines[i].strip())
                    i += 1
            story.append(Paragraph(ptext(" ".join(para)), styles["BodyText"]))
    doc = SimpleDocTemplate(str(HERE/(name+".pdf")), pagesize=(595,842),
                            leftMargin=56, rightMargin=56, topMargin=48, bottomMargin=48,
                            title=name, author="Physics validation / neural-holography-mini")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Rendered {name}: {len(eq_paths)} equations, PDF + offline HTML", flush=True)


if __name__ == "__main__":
    render("physics_proof")
    render("results")
