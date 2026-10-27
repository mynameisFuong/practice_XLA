"""Build a print report from Markdown with embedded Unicode fonts.

Requires requirements-report.txt. On Windows/Linux pass --font-dir containing
Arial.ttf/Arial Bold.ttf, or install DejaVuSans fonts at a standard location.
"""
from pathlib import Path
import argparse,re
from xml.sax.saxutils import escape
import markdown
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,Preformatted

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font-dir');args=parser.parse_args()
    if args.font_dir:
        folder=Path(args.font_dir)
        files=[folder/'Arial.ttf',folder/'Arial Bold.ttf',folder/'Arial Italic.ttf']
    elif Path('/System/Library/Fonts/Supplemental/Arial.ttf').exists():
        folder=Path('/System/Library/Fonts/Supplemental')
        files=[folder/'Arial.ttf',folder/'Arial Bold.ttf',folder/'Arial Italic.ttf']
    else:
        folder=Path('/usr/share/fonts/truetype/dejavu')
        files=[folder/'DejaVuSans.ttf',folder/'DejaVuSans-Bold.ttf',folder/'DejaVuSans-Oblique.ttf']
    for name,file in zip(['Body','BodyBold','BodyItalic'],files):
        if not file.exists():raise FileNotFoundError(f'Unicode font missing: {file}; pass --font-dir')
        pdfmetrics.registerFont(TTFont(name,str(file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody',fontName='Body',fontSize=10.5,leading=15.5,spaceAfter=8,splitLongWords=True))
    styles.add(ParagraphStyle(name='ReportTitle',fontName='BodyBold',fontSize=20,leading=27,spaceAfter=17))
    styles.add(ParagraphStyle(name='ReportH2',fontName='BodyBold',fontSize=15,leading=20,spaceBefore=17,spaceAfter=10,keepWithNext=True))
    styles.add(ParagraphStyle(name='ReportH3',fontName='BodyBold',fontSize=12,leading=17,spaceBefore=11,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle(name='Cell',fontName='Body',fontSize=8.2,leading=11.7,spaceAfter=0))
    styles.add(ParagraphStyle(name='CodeReport',fontName='Body',fontSize=8,leading=11,spaceAfter=10))
    html=markdown.markdown((ROOT/'reports/BAO_CAO.md').read_text(encoding='utf-8'),extensions=['tables','fenced_code'])
    soup=BeautifulSoup(html,'html.parser')
    def inline(node):
        text=''.join(str(c) for c in node.contents)
        text=re.sub(r'<code>(.*?)</code>',r'<font name="Body">\1</font>',text,flags=re.S)
        text=text.replace('<strong>','<b>').replace('</strong>','</b>').replace('<em>','<i>').replace('</em>','</i>')
        return text
    story=[];usable_width=A4[0]-100
    for tag in soup.children:
        if not getattr(tag,'name',None):continue
        name=tag.name
        if name in ['h1','h2','h3']:
            style={'h1':'ReportTitle','h2':'ReportH2','h3':'ReportH3'}[name]
            story.append(Paragraph(inline(tag),styles[style]))
        elif name=='p':
            img=tag.find('img')
            if img:
                path=ROOT/'reports'/img['src']
                story.append(Image(str(path),width=usable_width,height=usable_width*.4));story.append(Spacer(1,8))
            else:story.append(Paragraph(inline(tag),styles['ReportBody']))
        elif name=='table':
            rows=[]
            for row in tag.find_all('tr'):
                rows.append([Paragraph(inline(c),styles['Cell']) for c in row.find_all(['th','td'])])
            n=len(rows[0])
            widths=[usable_width/n]*n
            if n==2:widths=[usable_width*.25,usable_width*.75]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d2d6db')),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef3')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,12)])
        elif name in ['ol','ul']:
            for i,li in enumerate(tag.find_all('li',recursive=False)):
                prefix=f'{i+1}. ' if name=='ol' else '- '
                story.append(Paragraph(prefix+inline(li),styles['ReportBody']))
        elif name=='pre':
            story.append(Preformatted(tag.get_text(),styles['CodeReport']))
    out=ROOT/'reports/BAO_CAO.pdf'
    def footer(canvas,doc):
        canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(50,29,'Đồ án cảnh báo nghi vấn trộm xe máy')
        canvas.drawRightString(A4[0]-50,29,f'Trang {doc.page}')
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=46,bottomMargin=48,
        title='Xây dựng hệ thống cảnh báo nghi vấn trộm xe máy ở hầm chung cư',author='Nhóm sinh viên')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)

if __name__=='__main__':main()
