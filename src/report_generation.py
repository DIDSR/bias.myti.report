from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Frame, PageTemplate, BaseDocTemplate, Flowable, PageBreak, Paragraph, Image
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from datetime import date, datetime


def on_page(canvas, doc, pagesize=letter):
    """
    Set up page header and footer.
    """
    page_num = canvas.getPageNumber()
    canvas.setFillColor(HexColor('#007CBA'))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(50, 40, "OSEL")
    canvas.setFillColor(HexColor('#444444'))
    canvas.drawCentredString(pagesize[0]/2-20, 40, "Accelerating patient access to innovative, safe, and effective medical devices through best-in-the-world regulatory science")
    canvas.drawCentredString(pagesize[0]-50, 40, f"Page {str(page_num)}")
    canvas.drawImage("UI_assets/fda_logo.jpg", pagesize[0]-90, pagesize[1]-50, 0.5*inch, 0.6*inch, preserveAspectRatio=True)

def create_report(metrics, img_path, study_type, exp_type, save_path):
    """
    Create and save the report.
    
    Arguments
    =========
    metrics
        list contains metrics to save in the report
    img_path
        list contains paths for the saved figures
    study_type
        string to indicate study type
    exp_type
        string to indicate bias amplification type
    save_path
        path to save the final report
        
    """
    # frame padding style
    padding = dict(
    leftPadding=72, 
    rightPadding=72,
    topPadding=72,
    bottomPadding=48)    
    frame = Frame(0, 0, *letter, **padding)
    
    figure_1 = Image(img_path[0], width=6*inch, height=4.5*inch, kind='proportional')
    figure_2 = Image(img_path[1], width=6*inch, height=4.5*inch, kind='proportional')
    curr_time = datetime.now().strftime('%H:%M:%S')
    styleSheet = getSampleStyleSheet()
    styleSheet.add(ParagraphStyle(name='Heading1_Center',
                          parent=styleSheet['Heading1'],
                          alignment=TA_CENTER))
    # adding report contents
    story = []
    # adding study title and descriptions
    if study_type == "Compare Bias Mitigation Methods":
        report_title = "Bia Mitigation Comparison Report"
        study_title = "Study: Bias Mitigation Method Comparison"
        study_desc = "The study is designed to systematically evaluate bias mitigation methods implemented by the user. " + \
        f"After bias amplification through {exp_type.lower()}, mitigation methods are applied on these biased models and " + \
        "assessed by their effectiveness under different levels of bias."
        figue_desc_1 = "Figures below present results for bias mitigation comparison. The first subplot presents " + \
        "the amplified bias (without mitigation), while the rest subplots show results from different implemented mitigation methods."
    elif study_type == "Study Finite Sample Size Effect":
        report_title = "Bias Finite Sample Effect Report"
        study_title = "Study: Finite Sample Size Test"
        study_desc = f"The study is designed to explore the bias amplification effect by {exp_type.lower()} when sample size is limited. " + \
        "Bias amplification is applied to models trained with different training set size " + \
        "and the degrees to which the bias is promoted are compared across these models."
        figue_desc_1 = "Figures below present results for finite sample size study. " + \
        "Subplots in the figure present results with different sample sizes used for model training."
    else:
        report_title = "Bias Amplification Report"
        study_title = "Study: Bias Amplification Study"
        study_desc = f"The study is designed to sysmatically amplify model bias by {exp_type.lower()}. " + \
        "The approach can produce models with different degrees of bias, " + \
        "which facilitates a systematic evaluation on bias mitigation methods."
        figue_desc_1 = "Figures below show bias amplification results by {exp_type.lower()}."
    
    if exp_type == "Quantitative Misrepresentation":
        exp_title = "Bias Ampilification Approach: Quantitative Misrepresentation"
        exp_desc = "The quantitative misrepresentation approach applies data selection prior to training so that " + \
        "the disease prevalence is different for different patient subgroups. Additional controls over the degree " + \
        "to which bias is amplified is taken by the amount of prevalence difference between subgroups."
        figue_desc_2 = "For these experiments, the positive-associated subgroup refers to the subgroup " + \
        "with the higher disease prevalence in the training set. The x-axis indicates the subgroup disease " + \
        "prevelance difference in the training set, while B indicates the baseline model."
    else:
        exp_title = "Bias Ampilification Approach: Inductive Transfer Learning"
        exp_desc = "The inductive transfer learning approach applies a two-step transfer learning approach " + \
        "where the AI is trained to classify patient attributes during the first step. " + \
        "The AI is then fine-tuned to perform clinical tasks during the second step. " + \
        "Additional controls over the degree to which bias is amplified is taken by number of frozen layers during fine-tuning in second step."
        figue_desc_2 = "For these experiments, the positive-associated subgroup refers to the subgroup associated " + \
        "with the same model output during extra transfer learning step. " + \
        "The x-axis indicates the number of layers being frozen during the final model fine-tune step, while B indicates the baseline model." 
    story.append(Paragraph(report_title + "<br/><br/>", styleSheet['Heading1_Center']))
    story.append(Paragraph(study_title, styleSheet['Heading2']))
    story.append(Paragraph(study_desc, styleSheet['BodyText']))
    story.append(Paragraph(exp_title, styleSheet['Heading2']))
    story.append(Paragraph(exp_desc, styleSheet['BodyText']))
    story.append(Paragraph("Results", styleSheet['Heading2']))
    story.append(Paragraph(figue_desc_1 + figue_desc_2, styleSheet['BodyText']))
    # adding figures
    for i, m in enumerate(metrics):
        story.append(Paragraph(m, styleSheet['Heading3']))
        figure = Image(img_path[i], width=6*inch, height=4.5*inch, kind='proportional')
        story.append(figure)
        if i != len(metrics)-1:
            story.append(PageBreak())
    # adding report metadata
    story.append(Paragraph("<br/>Report Metadata", styleSheet['Heading2']))
    story.append(Paragraph("Report generated by: myti.report v1.0", styleSheet['BodyText']))
    story.append(Paragraph(f"Date: {date.today()}", styleSheet['BodyText']))
    story.append(Paragraph(f"Time: {curr_time}", styleSheet['BodyText']))
    # establish a document
    doc = BaseDocTemplate(save_path, pagesize=letter)
    # creating a page template
    frontpage = PageTemplate(id='FrontPage',
                             frames=[frame],
                             onPage=on_page,
                             pagesize=letter)
    # adding the story to the template and template to the document
    doc.addPageTemplates(frontpage)
    # building the report
    doc.build(story)

if __name__ == "__main__":
    m = ['AUROC', 'Predicted Prevalence']
    img_path = ['../example/figure_predicted prevalence.png', '../example/figure_AUROC.png']
    study = "Compare Bias Mitigation Methods"
    exp = "Inductive Transfer Learning"
    save_path = 'Example_output.pdf'
    create_report(m, img_path, study, exp, save_path)