# -*- coding: utf-8 -*-
"""
Updates the DA3 Evaluation Form, Project Report, and Presentation PPT with:
- Row 1: 24BAI1287 - Ankit Sthitapranjya (Role: Bandit Policy Modeling & Commercial Guardrails)
- Row 2: 24BAI1428 - Ananya Shandilya (Role: Data Pipeline, Demand Simulator & Streamlit UI)
- Rows 3 and 4 removed completely.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from pptx import Presentation
from pptx.util import Inches as PptInches, Pt as PptPt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor as PptRGBColor

def update_form():
    print("Updating reports/DA3_Project_Review_Evaluation_Form.docx...")
    doc = docx.Document()

    # Narrow 0.8 inch margins to guarantee 1 single page
    for s in doc.sections:
        s.top_margin = Inches(0.7)
        s.bottom_margin = Inches(0.7)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # Top Header Table (2 columns)
    top_table = doc.add_table(rows=2, cols=2)
    top_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    top_table.autofit = False
    top_table.columns[0].width = Inches(4.3)
    top_table.columns[1].width = Inches(2.7)

    c00 = top_table.cell(0, 0)
    c00.text = "Code: BCSE209L/Subject: Machine Learning"
    c00.paragraphs[0].runs[0].bold = True
    c00.paragraphs[0].runs[0].font.size = Pt(10.5)

    c01 = top_table.cell(0, 1)
    c01.text = "Class: CH2026270100894 /Slot: B2"
    c01.paragraphs[0].runs[0].bold = True
    c01.paragraphs[0].runs[0].font.size = Pt(10.5)

    c10 = top_table.cell(1, 0)
    c10.text = "Faculty: S. Sridevi, Associate Professor"
    c10.paragraphs[0].runs[0].bold = True
    c10.paragraphs[0].runs[0].font.size = Pt(10.5)

    c11 = top_table.cell(1, 1)
    c11.text = "School: CNI"
    c11.paragraphs[0].runs[0].bold = True
    c11.paragraphs[0].runs[0].font.size = Pt(10.5)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(14)
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
    doc.add_paragraph().paragraph_format.space_before = Pt(10)

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

    # Project Team No & Title
    doc.add_paragraph().paragraph_format.space_before = Pt(8)

    p_team = doc.add_paragraph()
    p_team.paragraph_format.space_after = Pt(6)
    r_team = p_team.add_run("Project Team No: ")
    r_team.bold = True
    r_team.font.size = Pt(10.5)
    p_team.add_run("________________________")

    p_title2 = doc.add_paragraph()
    p_title2.paragraph_format.space_after = Pt(12)
    r_title2 = p_title2.add_run("Project Title: ")
    r_title2.bold = True
    r_title2.font.size = Pt(10.5)
    r_val = p_title2.add_run("Personalized Dynamic Discount Recommendation Using Contextual Multi-Armed Bandits")
    r_val.font.size = Pt(10.5)

    # Table 2: Exactly 2 rows (Rows 3 and 4 removed!)
    t2 = doc.add_table(rows=3, cols=7) # Header + 2 student rows
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False

    widths2 = [Inches(0.55), Inches(1.2), Inches(1.6), Inches(1.75), Inches(0.9), Inches(0.85), Inches(0.85)]
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

    students = [
        ("1", "24BAI1287", "Ankit Sthitapranjya", "Bandit Policy Modeling (LinUCB & TS), Sherman-Morrison Update & Commercial Guardrails"),
        ("2", "24BAI1428", "Ananya Shandilya", "Data Preprocessing Pipeline, Retail Demand Simulator Calibration & Streamlit UI Dashboard")
    ]

    for r_idx, (sno, reg, name, role) in enumerate(students, start=1):
        row = t2.rows[r_idx]
        
        # S.No
        c_sno = row.cells[0]
        c_sno.text = sno
        c_sno.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_sno.paragraphs[0].runs[0].font.size = Pt(9.5)

        # Reg No
        c_reg = row.cells[1]
        c_reg.text = reg
        c_reg.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_reg.paragraphs[0].runs[0].bold = True
        c_reg.paragraphs[0].runs[0].font.size = Pt(9.5)

        # Name
        c_name = row.cells[2]
        c_name.text = name
        c_name.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_name.paragraphs[0].runs[0].bold = True
        c_name.paragraphs[0].runs[0].font.size = Pt(9.5)

        # Role
        c_role = row.cells[3]
        c_role.text = role
        c_role.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        c_role.paragraphs[0].runs[0].font.size = Pt(8.5)

        # Remaining cells (Report format, Rubric 1, Rubric 2) are left blank for faculty marks

    # Borders and cell padding for Table 2
    for row in t2.rows:
        trPr = row._tr.get_or_add_trPr()
        trHeight = parse_xml(r'<w:trHeight {} w:val="500" w:hRule="atLeast"/>'.format(nsdecls('w')))
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
    print("  -> Saved reports/DA3_Project_Review_Evaluation_Form.docx (Clean 2-member team, 1-page)")

def update_project_report_metadata():
    # BCSE209L_DA3_Project_Report.docx is now comprehensively generated by build_comprehensive_da3_report.py
    # which already includes the exact team metadata table.
    pass

def update_presentation_metadata():
    print("Updating team names in slides/DA3_Presentation.pptx...")
    prs = Presentation("slides/DA3_Presentation.pptx")
    slide1 = prs.slides[0]

    for shape in slide1.shapes:
        if shape.has_text_frame:
            tf = shape.text_frame
            for p in tf.paragraphs:
                if "Evaluation Dates:" in p.text:
                    p.text = "Course: BCSE209L — Machine Learning | Digital Assessment 3 (DA3 Project Review)\n" \
                             "Class: CH2026270100894 | Slot: B2 | School: CNI\n" \
                             "Faculty: Dr. S. Sridevi, Associate Professor\n" \
                             "Team Members: Ankit Sthitapranjya (24BAI1287) & Ananya Shandilya (24BAI1428)\n" \
                             "Evaluation Dates: 15.10.2026 to 19.10.2026 | Marks Allocated: 10 Marks"
                    p.font.size = PptPt(13)
                    p.font.color.rgb = PptRGBColor(71, 85, 105)

    prs.save("slides/DA3_Presentation.pptx")
    print("  -> Saved slides/DA3_Presentation.pptx with team metadata.")

if __name__ == "__main__":
    update_form()
    update_project_report_metadata()
    update_presentation_metadata()
    print("All team updates completed successfully.")
