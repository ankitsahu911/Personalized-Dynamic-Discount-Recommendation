# -*- coding: utf-8 -*-
"""
Recreates the EXACT 1-page DA3 Project Review Evaluation Form from the faculty image:
media_1792048040673_2f965bd2.png
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def generate_exact_form():
    doc = docx.Document()
    
    # 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # Top Header (2 columns using a borderless table or spaced tabs)
    top_table = doc.add_table(rows=2, cols=2)
    top_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    top_table.autofit = False

    # Widths
    top_table.columns[0].width = Inches(4.2)
    top_table.columns[1].width = Inches(2.8)

    # Row 0
    c00 = top_table.cell(0, 0)
    c00.text = "Code: BCSE209L/Subject: Machine Learning"
    c00.paragraphs[0].runs[0].bold = True
    c00.paragraphs[0].runs[0].font.size = Pt(10.5)

    c01 = top_table.cell(0, 1)
    c01.text = "Class: CH2026270100894 /Slot: B2"
    c01.paragraphs[0].runs[0].bold = True
    c01.paragraphs[0].runs[0].font.size = Pt(10.5)

    # Row 1
    c10 = top_table.cell(1, 0)
    c10.text = "Faculty: S. Sridevi, Associate Professor"
    c10.paragraphs[0].runs[0].bold = True
    c10.paragraphs[0].runs[0].font.size = Pt(10.5)

    c11 = top_table.cell(1, 1)
    c11.text = "School: CNI"
    c11.paragraphs[0].runs[0].bold = True
    c11.paragraphs[0].runs[0].font.size = Pt(10.5)

    # Spacing
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(14)
    p_space.paragraph_format.space_after = Pt(10)

    # Form Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(14)
    run_t = p_title.add_run("DA3 Project Review Evaluation Form")
    run_t.bold = True
    run_t.font.size = Pt(12)

    # Table 1: Dates & Marks
    t1 = doc.add_table(rows=2, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False

    widths1 = [Inches(0.8), Inches(2.2), Inches(2.6), Inches(1.4)]
    for row in t1.rows:
        for idx, width in enumerate(widths1):
            row.cells[idx].width = width

    headers1 = ["S.No", "Description", "Presentation &\nDemonstration Dates", "Mark allocated"]
    for i, h in enumerate(headers1):
        cell = t1.cell(0, i)
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].bold = True
        p.runs[0].font.size = Pt(10)

    row1_vals = ["1", "DA3", "15.10.2026 to 19.10.2026", "10"]
    for i, val in enumerate(row1_vals):
        cell = t1.cell(1, i)
        cell.text = val
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].bold = True
        p.runs[0].font.size = Pt(10)

    # Add borders to Table 1
    for row in t1.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = parse_xml(r'''
                <w:tcBorders {} >
                    <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                    <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                    <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                    <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                </w:tcBorders>
            '''.format(nsdecls('w')))
            tcPr.append(tcBorders)

    # Body Paragraphs
    doc.add_paragraph().paragraph_format.space_before = Pt(12)

    p_eval = doc.add_paragraph()
    p_eval.paragraph_format.space_after = Pt(2)
    r = p_eval.add_run("Project evaluation of DA3:")
    r.bold = True
    r.font.size = Pt(10.5)

    p_desc = doc.add_paragraph()
    p_desc.paragraph_format.space_after = Pt(2)
    p_desc.paragraph_format.line_spacing = 1.15
    r = p_desc.add_run("A Project Report submitted in msteams as per template and Hardware/Software demonstration shown for evaluation.\n"
                       "Submission already made in msteams\n"
                       "Hardware/Software Demo video link submitted in msteams: Demonstration of 1.5 to 2 minutes duration (This is for file submission)")
    r.font.size = Pt(10)

    # Team & Title
    doc.add_paragraph().paragraph_format.space_before = Pt(8)

    p_team = doc.add_paragraph()
    p_team.paragraph_format.space_after = Pt(8)
    r_team = p_team.add_run("Project Team No: ")
    r_team.bold = True
    r_team.font.size = Pt(10.5)

    p_title2 = doc.add_paragraph()
    p_title2.paragraph_format.space_after = Pt(14)
    r_title2 = p_title2.add_run("Project Title: ")
    r_title2.bold = True
    r_title2.font.size = Pt(10.5)
    # Pre-fill project title
    r_val = p_title2.add_run("Personalized Dynamic Discount Recommendation Using Contextual Multi-Armed Bandits")
    r_val.font.size = Pt(10.5)

    # Table 2: Student Details & Marks
    t2 = doc.add_table(rows=5, cols=7)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False

    widths2 = [Inches(0.6), Inches(1.3), Inches(1.5), Inches(0.9), Inches(1.1), Inches(0.9), Inches(0.9)]
    for row in t2.rows:
        for idx, width in enumerate(widths2):
            row.cells[idx].width = width

    headers2 = ["S.No", "Team\nmembers\nReg.No", "Team members\nName", "Role", "Report as\nper format", "Rubric 1", "Rubric 2"]
    for i, h in enumerate(headers2):
        cell = t2.cell(0, i)
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].bold = True
        p.runs[0].font.size = Pt(9.5)

    for r_idx in range(1, 5):
        cell_sno = t2.cell(r_idx, 0)
        cell_sno.text = str(r_idx)
        cell_sno.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell_sno.paragraphs[0].runs[0].font.size = Pt(9.5)

    # Add borders and height to Table 2
    for row in t2.rows:
        trPr = row._tr.get_or_add_trPr()
        trHeight = parse_xml(r'<w:trHeight {} w:val="380" w:hRule="atLeast"/>'.format(nsdecls('w')))
        trPr.append(trHeight)
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = parse_xml(r'''
                <w:tcBorders {} >
                    <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                    <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                    <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                    <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>
                </w:tcBorders>
            '''.format(nsdecls('w')))
            tcPr.append(tcBorders)

    doc.save("reports/DA3_Project_Review_Evaluation_Form.docx")
    print("Exact 1-page DA3_Project_Review_Evaluation_Form.docx created successfully.")

if __name__ == "__main__":
    generate_exact_form()
