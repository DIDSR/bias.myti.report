"""
Main design script for RST related to bias project
"""
import pandas as pd
from PyQt6.QtWidgets import * 
from PyQt6.QtCore import *
from PyQt6.QtGui import * 
from PyQt6.QtGui import QPixmap
import sys 
import shutil

from plot_generation import *

class InitialPage(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Myti.Report')
        self.csv_path = '../example/example.csv'
        self.exp_type = 'Direct Approach'
        self.sample_size = False
        self.miti_compare = False
        self.UIComponents()


    def UIComponents(self):
    
        # # csv file upload part
        # #
        # # file indicating label
        self.lbl_up_load_file = QLabel('Uploaded Input File', self)
        self.lbl_up_load_file.move(100, 55)
        
        # # edit box for file directory   
        self.edit_up_load_file = QLineEdit('../example/example.csv', self)     
        self.edit_up_load_file.resize(350,25)
        self.edit_up_load_file.move(240,50)

        # # button to browse the csv file  
        self.btn_up_load_file = QPushButton('Browse', self)      
        self.btn_up_load_file.setToolTip('Select <b>CSV File</b> from folder')
        self.btn_up_load_file.resize(self.btn_up_load_file.sizeHint())
        self.btn_up_load_file.move(620, 50)
        self.btn_up_load_file.clicked.connect(self.upload_csv)
        
        # # experiment setting part
        # #
        # # Experiment explanation 
        self.lbl_exp_title = QLabel('Description of the approach', self)       
        self.lbl_exp_title.move(400, 120)
        self.lbl_exp_dscp = QLabel('Please select an experiment type.', self)
        self.lbl_exp_dscp.setStyleSheet("border : 2px solid black;") 
        self.lbl_exp_dscp.setGeometry(400, 160, 300, 250)
        self.lbl_exp_dscp.setWordWrap(True) 
        # # experimeny type indicating label
        self.lbl_exp_type = QLabel('Select Experiment Type', self)        
        self.lbl_exp_type.move(100, 120)
        # # direct or indirect selection menu  
        self.combo_exp_type = QComboBox(self)      
        self.combo_exp_type.addItem('-Please Select-')
        self.combo_exp_type.addItems(['Direct Approach','Indirect Approach'])
        self.combo_exp_type.move(100, 170)
        self.combo_exp_type.currentTextChanged.connect(self.approach_type)
        # # finite sample size checkbox          
        self.cb_sample_size = QCheckBox('Finite Sample Size', self)      
        self.cb_sample_size.move(100, 220)
        self.cb_sample_size.toggled.connect(self.sample_size_check) 
        sample_size_info = 'If finite sample size is selected, results using different training dataset size will be computed and visualized.'
        self.lbl_sample_size = QLabel(sample_size_info, self)
        self.lbl_sample_size.setGeometry(100, 240, 250, 100)
        self.lbl_sample_size.setWordWrap(True) 
        # # mitigation method compare checkbox
        self.cb_miti_compare = QCheckBox('Compare Bias Mitigation Methods', self)        
        self.cb_miti_compare.move(100, 330)
        self.cb_miti_compare.toggled.connect(self.miti_compare_check) 
        miti_compare_info = 'If comparing bias mitigation methods is selected, results using different bias mitigation methods will be campared and visualized.'
        self.lbl_miti_compare = QLabel(sample_size_info, self)
        self.lbl_miti_compare.setGeometry(100, 350, 250, 100)
        self.lbl_miti_compare.setWordWrap(True) 
                        
        # # button for next page    
        self.btn_next_page = QPushButton('Next Page', self)    
        self.btn_next_page.resize(self.btn_next_page.sizeHint())
        self.btn_next_page.move(600, 550)
        self.btn_next_page.clicked.connect(self.next_page)

    def upload_csv(self):
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(None, "Import CSV", "", "CSV data files (*.csv)")
        self.edit_up_load_file.setText(fname[0])
        self.csv_path = fname[0]
        
    def approach_type(self):
        text = str(self.combo_exp_type.currentText())
        global exp_type
        self.exp_type = text 
        if text == 'Direct Approach':
            info = 'Direct approach applies data selection prior to training so that the disease prevalence is different for different patient subgroups.\n' + \
            '\nThe degree to which bias is promoted can be controlled by changing the degree of prevalence gap between subgroups.'
        elif text == 'Indirect Approach':
            info = 'Indirect approach applies an extra step of transfer learning prior to target model fine-tuning to promote spurious correlation between model task and subgroup infomation.\n' + \
            '\nThe degree to which bias is promoted can be controlled by number of frozen layers in last step fine-tuning.'
        else:
            info = 'Please select an experiment type.'       
        self.lbl_exp_dscp.setText(info)
    
    def sample_size_check(self):
        self.sample_size = True if self.cb_sample_size.isChecked() else False
        
    def miti_compare_check(self):
        self.miti_compare = True if self.cb_miti_compare.isChecked() else False        
    
    def next_page(self):
        secondpage = SecondPage(self.csv_path, self.exp_type, self.sample_size, self.miti_compare) 
        widget.addWidget(secondpage)   # adding second page
        widget.setCurrentWidget(secondpage)


class SecondPage(QWidget):

    def __init__(self, csv_path, exp_type, sample_size=False, miti_compare=False):
        super().__init__()
        self.setWindowTitle('CSV Information')
        self.csv_path = csv_path
        self.exp_type = exp_type
        self.sample_size = sample_size
        self.miti_compare = miti_compare
        self.UIComponents()
        
    def UIComponents(self):
       
        # # Title Label
        self.lbl_title = QLabel('Indicate Columns', self)
        self.lbl_title.move(100, 50)
        
        column_list = self.get_columns()    
        # # subgroup label selection
        self.lbl_subg = QLabel('Subgroup label:', self)
        self.lbl_subg.move(100, 100)
        self.combo_subg = QComboBox(self)
        self.combo_subg.addItem('Subgroup')
        self.combo_subg.addItems(column_list)
        self.combo_subg.move(300, 100)
        # # plot metric selection
        self.lbl_metric = QLabel('Metric for plot:', self)
        self.lbl_metric.move(100, 150)
        self.combo_metric = QComboBox(self)
        self.combo_metric.addItem('Sensitivity')
        self.combo_metric.addItems(column_list)
        self.combo_metric.move(300, 150)
        # # prevalence selection
        self.lbl_exp_1 = QLabel('Prevalence 1:', self)
        self.lbl_exp_1.move(100, 200)
        self.combo_exp_1 = QComboBox(self)
        self.combo_exp_1.addItem('Prevalence F')
        self.combo_exp_1.addItems(column_list)
        self.combo_exp_1.move(300, 200)
        self.lbl_exp_2 = QLabel('Prevalence 2:', self)
        self.lbl_exp_2.move(100, 250)
        self.combo_exp_2 = QComboBox(self)
        self.combo_exp_2.addItem('Prevalence M')
        self.combo_exp_2.addItems(column_list)
        self.combo_exp_2.move(300, 250)

        
        # # button to the previous page
        self.btn_prev_page = QPushButton('Previous Page', self)
        self.btn_prev_page.resize(self.btn_prev_page.sizeHint())
        self.btn_prev_page.move(500, 550)
        self.btn_prev_page.clicked.connect(self.prev_page)
        # # button to the next page
        self.btn_next_page = QPushButton('Next Page', self)
        self.btn_next_page.resize(self.btn_next_page.sizeHint())
        self.btn_next_page.move(600, 550)
        self.btn_next_page.clicked.connect(self.next_page)

    def get_columns(self):
        data = pd.read_csv(self.csv_path)
        return list(data.columns)

    def next_page(self):        
        result_plotting(str(self.combo_subg.currentText()), str(self.combo_metric.currentText()), 
        str(self.combo_exp_1.currentText()), str(self.combo_exp_2.currentText()), self.csv_path)
        thirdpage = FinalPage() 
        widget.addWidget(thirdpage)   # adding last page
        widget.setCurrentWidget(thirdpage)    
        
    def prev_page(self):
        widget.setCurrentWidget(firstpage)
        
class FinalPage(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Myti Results')
        self.current_plot = 0
        self.UIComponents()
        
        
    def UIComponents(self):
        
        # # title for plot selection
        self.lbl_option = QLabel('Plot Options', self)
        self.lbl_option.move(100, 20)
 
        # # adding example image
        self.fig_1_label = QLabel(self)
        self.pixmap_1 = QPixmap('../example/example_1.png')
        self.fig_1_label.setPixmap(self.pixmap_1.scaled(200,150))
        self.fig_1_label.resize(200,150)
        self.fig_1_label.move(100,40)        
        self.fig_2_label = QLabel(self)
        self.pixmap_2 = QPixmap('../example/example_2.png')
        self.fig_2_label.setPixmap(self.pixmap_2.scaled(200,150))
        self.fig_2_label.resize(200,150)
        self.fig_2_label.move(100,210)
        
        self.fig_3_label = QLabel(self)
        self.pixmap_3 = QPixmap('../example/example_3.png')
        self.fig_3_label.setPixmap(self.pixmap_3.scaled(200,150))
        self.fig_3_label.resize(200,150)
        self.fig_3_label.move(100,380)
                
        # # title for displaying plot
        self.lbl_plot = QLabel('Selected Plot', self)
        self.lbl_plot.move(350, 20)
        # # title for plot discription
        self.lbl_dscp = QLabel('Plot Description', self)
        self.lbl_dscp.move(350, 320)
        # # position for selected plot
        self.lbl_selected_plot = QLabel(self)
        self.lbl_selected_plot.resize(360,270)
        self.lbl_selected_plot.move(350,40)
        # # description for the plot
        self.lbl_selected_dscp = QLabel(self)
        self.lbl_selected_dscp.setStyleSheet("border : 2px solid black;")
        self.lbl_selected_dscp.resize(360,200)
        self.lbl_selected_dscp.move(350,340)
        self.lbl_selected_dscp.setWordWrap(True) 
        # # button to save figure
        self.btn_save_fig = QPushButton('Save Figure', self)
        self.btn_save_fig.resize(self.btn_save_fig.sizeHint())
        self.btn_save_fig.move(600, 15)
        self.btn_save_fig.clicked.connect(self.save_fig)
        
        # # events when clicking on plot options        
        self.fig_1_label.mousePressEvent = self.fig_1_click
        self.fig_2_label.mousePressEvent = self.fig_2_click
        self.fig_3_label.mousePressEvent = self.fig_3_click
        # # button to the previous page
        self.btn_prev_page = QPushButton('Previous Page', self)
        self.btn_prev_page.resize(self.btn_prev_page.sizeHint())
        self.btn_prev_page.move(500, 550)
        self.btn_prev_page.clicked.connect(self.prev_page)
        # # button to quit
        self.btn_quit_page = QPushButton('Quit', self)
        self.btn_quit_page.resize(self.btn_quit_page.sizeHint())
        self.btn_quit_page.move(600, 550)
        self.btn_quit_page.clicked.connect(self.quit_page)


    def fig_1_click(self, event):
        self.fig_1_label.setStyleSheet("border : 4px solid Green;")
        self.fig_2_label.setStyleSheet("border : 0px solid Black;")
        self.fig_3_label.setStyleSheet("border : 0px solid Black;")
        self.lbl_selected_plot.setPixmap(self.pixmap_1.scaled(360,270))
        info = open('../example/tmp/description_1.txt').read()
        self.lbl_selected_dscp.setText(info)
        self.current_plot = 1       
        
    def fig_2_click(self, event):
        self.fig_2_label.setStyleSheet("border : 4px solid Green;")
        self.fig_1_label.setStyleSheet("border : 0px solid Black;")
        self.fig_3_label.setStyleSheet("border : 0px solid Black;") 
        self.lbl_selected_plot.setPixmap(self.pixmap_2.scaled(360,270))
        info = open('../example/tmp/description_2.txt').read()
        self.lbl_selected_dscp.setText(info)
        self.current_plot = 2   
        
    def fig_3_click(self, event):
        self.fig_3_label.setStyleSheet("border : 4px solid Green;")
        self.fig_2_label.setStyleSheet("border : 0px solid Black;")
        self.fig_1_label.setStyleSheet("border : 0px solid Black;")
        self.lbl_selected_plot.setPixmap(self.pixmap_3.scaled(360,270))
        info = open('../example/tmp/description_3.txt').read()
        self.lbl_selected_dscp.setText(info)
        self.current_plot = 3
        
    def prev_page(self):
        widget.setCurrentWidget(secondpage)
    
    def quit_page(self):
        shutil.rmtree('../example/tmp/')
        widget.close()
    
    def save_fig(self):
        name = QFileDialog.getSaveFileName(self, 'Save File',"PNG (*.png)")
        shutil.copy(f'../example/tmp/fig_text_{self.current_plot}.png', name[0])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    firstpage = InitialPage()
    widget.addWidget(firstpage)   # create an instance of the first page class and add it to stackedwidget   
    widget.setFixedHeight(600)
    widget.setFixedWidth(800)
    widget.setCurrentWidget(firstpage)   # setting the page that you want to load when application starts up. you can also use setCurrentIndex(int)
    widget.show()
    
    sys.exit(app.exec())
