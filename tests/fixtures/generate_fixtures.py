#!/usr/bin/env python
"""
tests/fixtures/generate_fixtures.py
────────────────────────────────────
Generates the binary test fixtures used by the scraper unit tests.

Run once to (re-)create the PDF fixture:
    python tests/fixtures/generate_fixtures.py

Requires: fpdf2  (already in requirements.txt)
"""
import os

from fpdf import FPDF
from fpdf.enums import XPos, YPos


def generate_fee_pdf(output_path: str) -> None:
    """Create a minimal PDF containing a fee structure table."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", size=16)
    # Use new_x/new_y instead of deprecated ln=True
    pdf.cell(0, 10, "WCE Fee Structure 2025-26",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0, 7,
        "Walchand College of Engineering, Sangli\n"
        "Official Fee Structure for Academic Year 2025-26\n"
        "(As approved by the Fee Regulating Authority, Maharashtra)",
        align="C",
    )
    pdf.ln(6)

    # Table header — use ASCII-safe currency notation (fpdf2 Helvetica is latin-1)
    pdf.set_font("Helvetica", "B", size=11)
    col_widths = [60, 55, 40, 35]
    headers = ["Category", "Tuition Fee (Rs.)", "Dev. Fee (Rs.)", "Total (Rs.)"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 9, h, border=1, align="C")
    pdf.ln()

    # Table rows
    rows = [
        ("General / OBC",              "Rs. 1,09,500", "Rs. 5,000",  "Rs. 1,14,500"),
        ("SC / ST (Govt. scholarship)", "Nil",          "Rs. 5,000",  "Rs. 5,000"),
        ("NRI / Foreign National",      "Rs. 2,70,000", "Rs. 10,000", "Rs. 2,80,000"),
        ("PWD (Differently Abled)",     "Nil",          "Rs. 5,000",  "Rs. 5,000"),
    ]
    pdf.set_font("Helvetica", size=10)
    for row in rows:
        for w, val in zip(col_widths, row):
            pdf.cell(w, 8, val, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", size=9)
    pdf.multi_cell(
        0, 6,
        "Note: The above fees are per academic year. Examination fee of Rs. 2,000 per "
        "semester is charged separately. Hostel fee (Rs. 45,000/year) is optional. "
        "SC/ST fee waiver is subject to submission of valid caste and income certificate. "
        "For queries, contact: accounts@walchandsangli.ac.in",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate_fee_pdf(os.path.join(here, "fee_structure.pdf"))
