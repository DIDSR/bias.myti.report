"""
Main design script for RST related to bias project
"""

from PyQt6.QtWidgets import * 
from PyQt6 import QtCore 
from PyQt6.QtGui import * 
from PyQt6.QtGui import QPixmap
import sys 


class InitialPage(QWidget):

    def __init__(self):
        super().__init__()

        self.UIComponents()


    def UIComponents(self):

        #self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle('Myti.Report')
        
        # # csv file upload part
        # #
        # # file indicating label
        lbl_up_load_file = QLabel('Uploaded Input File', self)
        #lbl_up_load_file.resize(30, 100)
        lbl_up_load_file.move(100, 55)
        
        # # edit box for file directory
        edit_up_load_file = QLineEdit('testing', self)
        edit_up_load_file.resize(350,25)
        edit_up_load_file.move(240,50)

        # # button to browse the csv file
        btn_up_load_file = QPushButton('Browse', self)
        btn_up_load_file.setToolTip('Select <b>CSV File</b> from folder')
        btn_up_load_file.resize(btn_up_load_file.sizeHint())
        btn_up_load_file.move(620, 50)
        
        # # experiment setting part
        # #
        # # experimeny type indicating label
        lbl_exp_type = QLabel('Select Experiment Type', self)
        lbl_exp_type.move(100, 200)
        # # direct or indirect selection menu
        combo_exp_type = QComboBox(self)
        combo_exp_type.addItem('Direct Approach')
        combo_exp_type.addItem('Indirect Approach')
        combo_exp_type.move(100, 250)
        # # finite sample size checkbox
        cb_sample_size = QCheckBox('Finite Sample Size', self)
        cb_sample_size.move(100, 300)
        # # mitigation method compare checkbox
        cb_myti_compare = QCheckBox('Compare Bias Mitigation Methods', self)
        cb_myti_compare.move(100, 350)
                
        # # Experiment explanation
        # #
        lbl_exp_title = QLabel('Description of the approach', self)
        lbl_exp_title.move(500, 200)
        
        # # button for next page
        btn_next_page = QPushButton('Next Page', self)
        btn_next_page.resize(btn_next_page.sizeHint())
        btn_next_page.move(600, 500)
        btn_next_page.clicked.connect(self.on_btn_next_page_clicked)


    def on_btn_next_page_clicked(self):
        widget.setCurrentWidget(secondpage)


class SecondPage(QWidget):

    def __init__(self):
        super().__init__()

        self.UIComponents()
        
    def UIComponents(self):

        #self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle('CSV Information')
        
        # # Title Label
        lbl_title = QLabel('Indicate Columns', self)
        lbl_title.move(100, 50)
        
        # # patient id selection
        lbl_patient_id = QLabel('Patient Identifier:', self)
        lbl_patient_id.move(100, 100)
        combo_patient_id = QComboBox(self)
        combo_patient_id.addItem('patient_id')
        combo_patient_id.move(300, 100)
        # # subgroup label selection
        lbl_subgroup = QLabel('Subgroup Label:', self)
        lbl_subgroup.move(100, 150)
        combo_subgroup = QComboBox(self)
        combo_subgroup.addItem('patient_sex')
        combo_subgroup.move(300, 150)
        # # output score selection
        lbl_output = QLabel('Model Output:', self)
        lbl_output.move(100, 200)
        combo_output = QComboBox(self)
        combo_output.addItem('output_socre')
        combo_output.move(300, 200)
        # # actual label selection
        lbl_label = QLabel('Sample Label:', self)
        lbl_label.move(100, 250)
        combo_label = QComboBox(self)
        combo_label.addItem('actual_class')
        combo_label.move(300, 250)
        # # experimeny type selection
        lbl_exp = QLabel('Experiment:', self)
        lbl_exp.move(100, 300)
        combo_exp = QComboBox(self)
        combo_exp.addItem('100_female_prev')
        combo_exp.move(300, 300)
        
        # # button to the previous page
        btn_prev_page = QPushButton('Previous Page', self)
        btn_prev_page.resize(btn_prev_page.sizeHint())
        btn_prev_page.move(500, 500)
        btn_prev_page.clicked.connect(self.on_btn_prev_page_clicked)
        # # button to the next page
        btn_next_page = QPushButton('Next Page', self)
        btn_next_page.resize(btn_next_page.sizeHint())
        btn_next_page.move(600, 500)
        btn_next_page.clicked.connect(self.on_btn_next_page_clicked)


    def on_btn_next_page_clicked(self):
        widget.setCurrentWidget(thirdpage)
        
    def on_btn_prev_page_clicked(self):
        widget.setCurrentWidget(firstpage)
        
class FinalPage(QWidget):

    def __init__(self):
        super().__init__()

        self.UIComponents()
        
    def UIComponents(self):

        #self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle('Myti Results')
        
        lbl_option = QLabel('Plot Options', self)
        lbl_option.move(100, 50)
        
        pixmap = QPixmap('image.png')
 
        # adding example image
        fig_1_label = QLabel(self)
        pixmap_1 = QPixmap('../example/example_1.png').scaled(200,160)
        fig_1_label.setPixmap(pixmap_1)
        fig_1_label.resize(200,160)
        fig_1_label.move(100,70)
        
        fig_2_label = QLabel(self)
        pixmap_2 = QPixmap('../example/example_2.png').scaled(200,160)
        fig_2_label.setPixmap(pixmap_2)
        fig_2_label.resize(200,160)
        fig_2_label.move(100,240)
        
        fig_3_label = QLabel(self)
        pixmap_3 = QPixmap('../example/example_3.png').scaled(200,160)
        fig_3_label.setPixmap(pixmap_3)
        fig_3_label.resize(200,160)
        fig_3_label.move(100,410)
        
        
        lbl_plot = QLabel('Selected Plot', self)
        lbl_plot.move(400, 50)
        
        lbl_dscp = QLabel('Plot Description', self)
        lbl_dscp.move(400, 300)
        
        btn_save_fig = QPushButton('Save Figure', self)
        btn_save_fig.resize(btn_save_fig.sizeHint())
        btn_save_fig.move(600, 50)
        
        btn_prev_page = QPushButton('Previous Page', self)
        btn_prev_page.resize(btn_prev_page.sizeHint())
        btn_prev_page.move(500, 500)
        btn_prev_page.clicked.connect(self.on_btn_prev_page_clicked)
        
        btn_quit_page = QPushButton('Quit', self)
        btn_quit_page.resize(btn_quit_page.sizeHint())
        btn_quit_page.move(600, 500)
        btn_quit_page.clicked.connect(QApplication.instance().quit)

    def on_btn_prev_page_clicked(self):
        widget.setCurrentWidget(secondpage)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    firstpage = InitialPage()
    widget.addWidget(firstpage)   # create an instance of the first page class and add it to stackedwidget
    secondpage = SecondPage() 
    widget.addWidget(secondpage)   # adding second page
    thirdpage = FinalPage() 
    widget.addWidget(thirdpage)   # adding last page    
    widget.setFixedHeight(600)
    widget.setFixedWidth(800)
    widget.setCurrentWidget(firstpage)   # setting the page that you want to load when application starts up. you can also use setCurrentIndex(int)
    widget.show()
    sys.exit(app.exec())
