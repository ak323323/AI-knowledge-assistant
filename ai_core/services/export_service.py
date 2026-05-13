from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document
from io import BytesIO


def generate_pdf(answer: str, sources: list):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("<b>AI Response</b>", styles["Heading1"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(answer, styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Sources</b>", styles["Heading2"]))

    for s in sources:
        content.append(Paragraph(s["source"], styles["Normal"]))

    doc.build(content)

    buffer.seek(0)
    return buffer


def generate_docx(answer: str, sources: list):
    doc = Document()

    doc.add_heading("AI Response", 0)
    doc.add_paragraph(answer)

    doc.add_heading("Sources", level=1)

    for s in sources:
        doc.add_paragraph(s["source"])

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return buffer