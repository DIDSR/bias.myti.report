"""
Main design script for RST related to bias project
"""
import pandas as pd
from PyQt6.QtWidgets import * 
from PyQt6.QtCore import *
from PyQt6.QtGui import * 
from PyQt6.QtGui import QPixmap
from PyQt6.QtSvgWidgets import *
import sys 
import shutil
from pathlib import Path
import os

from plot_generation import *

class ClickLabel(QLabel):
    """class to create clickable QLabel object"""
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        QLabel.mousePressEvent(self, event)


class Page(QWidget):
  """"""
  def __init__(self, parent, page_number):
    super().__init__()
    self.parent = parent
    self.page_number = page_number 
    self.setProperty("class", "page")  
    
  def UIComponents(self):
    """ Placeholder"""
    pass
    
  def load(self):
    """ Run background processes and load UI components, provided the conditions are met """
    if self.check_conditions():
      self.run_background()
      self.UIComponents()
      self.setStyleSheet(styleSheet)
      self.update()
      
  def check_conditions(self, *conditions): 
    """ Conditions are variables that must be input in order for the page to load.
    This is the default for page(s) with no conditions """
    return True
    
  def run_background(self):
    """ Runs any background work that needs to be run before the page can be loaded. 
    This is the default for page(s) with no background work. """
    pass

def clearLayout(layout):
  while layout.count():
    child = layout.takeAt(0)
    if child.widget():
      child.widget().deleteLater()
      
class InitialPage(Page):
    """Class for the first page (user input) of the tool."""
    def __init__(self, parent):
        super().__init__(parent=parent, page_number=1)
        # Set layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def UIComponents(self):
        """Widgets for the page."""
        # Set layout
        clearLayout(self.layout)
        
        # # program title
        self.title_box = QWidget()
        self.title_box.setObjectName("box")
        
        self.lbl_title = QLabel('bias.myti.Report', self.title_box)
        self.lbl_title.setObjectName("title")
        self.layout.addWidget(self.lbl_title)
        #self.lbl_title.setFont(QFont('Arial', 20))
        self.lbl_subtitle = QLabel('A tool to facilitate the comparison of user-implemented bias mitigation methods for AI models', self)
        self.lbl_subtitle.setObjectName("subtitle")
        self.layout.addWidget(self.lbl_subtitle)
         
        # # csv file upload part
        self.upload_box = QWidget()
        self.upload_layout = QHBoxLayout()
        # #
        # # file indicating label
        self.lbl_up_load_file = QLabel('Uploaded Input File', self)
        self.upload_layout.addWidget(self.lbl_up_load_file)
        
        # # edit box for file directory   
        self.edit_up_load_file = QLineEdit('../example/example.csv', self) 
        self.edit_up_load_file.resize(350,25)
        self.upload_layout.addWidget(self.edit_up_load_file)

        # # button to browse the csv file  
        self.btn_up_load_file = QPushButton('Browse', self)      
        self.btn_up_load_file.setToolTip('Select <b>CSV File</b> from folder')
        self.btn_up_load_file.resize(self.btn_up_load_file.sizeHint())
        self.upload_layout.addWidget(self.btn_up_load_file)
        self.btn_up_load_file.clicked.connect(self.upload_csv)
        
        self.upload_box.setLayout(self.upload_layout)
        self.layout.addWidget(self.upload_box)
        
        self.layout.addSpacing(3)

        # # experiment setting part
        self.exp_box = QWidget()
        self.exp_layout = QHBoxLayout()
        
        self.settings_box = QWidget()
        self.settings_layout = QGridLayout()
        self.settings_box.setLayout(self.settings_layout)
        
        self.description_box = QWidget()
        self.desc_layout = QGridLayout()
        self.description_box.setLayout(self.desc_layout)
         
        self.exp_layout.addWidget(self.description_box)
        self.exp_layout.addWidget(self.settings_box)
        self.exp_layout.setStretch(1,10)
        # #
        # # Experiment explanation 
        # # amplification type indicating label
        self.lbl_exp_type = QLabel('Select Amplification Type', self)  
        self.desc_layout.addWidget(self.lbl_exp_type, 0, 0, 1, 1)  
        # # direct or indirect selection menu  
        self.combo_exp_type = QComboBox(self)      
        self.combo_exp_type.addItem('-Please Select-')
        self.combo_exp_type.addItems(list(self.parent.experiments.keys()))
        self.desc_layout.addWidget(self.combo_exp_type, 0, 1, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        self.combo_exp_type.currentTextChanged.connect(self.approach_type)
        self.exp_descriptions = {}
        self.exp_layouts = {}
        for i, (exp, desc) in enumerate(self.parent.experiments.items()):
          self.exp_descriptions[exp] = QWidget()
          self.exp_layouts[exp] = QVBoxLayout()
          self.exp_descriptions[exp].setLayout(self.exp_layouts[exp])
          label = QLabel(exp)
          self.exp_layouts[exp].addWidget(label)
          dlabel = QLabel(desc)
          dlabel.setWordWrap(True)
          dlabel.resize(300, 250)
          self.exp_layouts[exp].addWidget(dlabel)
          self.exp_descriptions[exp].setObjectName("not_selected")
          self.desc_layout.addWidget(self.exp_descriptions[exp], i+1, 0, 1, 2)
        self.desc_layout.setRowStretch(3, 10)
        self.desc_layout.setColumnStretch(1, 2)
        
        
        # # study type selection menu
        self.lbl_stdy_type = QLabel('Select Study Type', self)  
        self.settings_layout.addWidget(self.lbl_stdy_type, 0, 0, 1, 1)   
        self.rb_sample_size = QRadioButton('Study Finite Sample Size Effect', self)            
        self.settings_layout.addWidget(self.rb_sample_size, 1, 0, 1, 2)  
        self.rb_sample_size.toggled.connect(self.study_update) 
        sample_size_info = 'If finite sample size is selected, results using different training dataset size will be computed and visualized.'
        self.lbl_sample_size = QLabel(sample_size_info, self)
        self.lbl_sample_size.resize(250, 100)
        self.settings_layout.addWidget(self.lbl_sample_size, 2, 0, 1, 2)
        self.lbl_sample_size.setWordWrap(True) 
        # # mitigation method compare checkbox
        self.rb_miti_compare = QRadioButton('Compare Bias Mitigation Methods', self)  
        self.settings_layout.addWidget(self.rb_miti_compare, 3, 0, 1, 2)      
        self.rb_miti_compare.toggled.connect(self.study_update) 
        miti_compare_info = 'If comparing bias mitigation methods is selected, results using different bias mitigation methods will be campared and visualized.'
        self.lbl_miti_compare = QLabel(miti_compare_info, self)
        self.lbl_miti_compare.resize(250, 100)
        self.lbl_miti_compare.setWordWrap(True) 
        self.settings_layout.addWidget(self.lbl_miti_compare, 4, 0, 1, 2)
        self.rb_none = QRadioButton('None', self)
        self.rb_none.setChecked(True)
        self.rb_none.toggled.connect(self.study_update)
        self.settings_layout.addWidget(self.rb_none, 5, 0, 1, 2)
        none_info = 'If none is selected, only results from bias amplification will be visualized.'
        self.lbl_none = QLabel(none_info, self)
        self.lbl_none.resize(250, 100)
        self.lbl_none.setWordWrap(True) 
        self.settings_layout.addWidget(self.lbl_none, 6, 0, 1, 2)
        self.settings_layout.setRowStretch(7, 100)
  
        self.exp_box.setLayout(self.exp_layout)
        self.layout.addWidget(self.exp_box)
        self.layout.addSpacing(1)
        # # set exp and study selection hidden initially
        self.description_box.setHidden(True)
        self.retain = QSizePolicy()
        self.retain.setRetainSizeWhenHidden(True)
        self.description_box.setSizePolicy(self.retain)
        self.settings_box.setHidden(True)
        self.settings_box.setSizePolicy(self.retain)


    def upload_csv(self):
        """Get the csv file from user input, and display amplification type selection"""
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(None, "Import CSV", "", "CSV data files (*.csv)")
        self.edit_up_load_file.setText(fname[0])
        self.parent.csv_path = fname[0]
        self.description_box.setHidden(False)

        
    def approach_type(self):
        """Update amplification type selected by user"""
        text = str(self.combo_exp_type.currentText())
        self.parent.exp_type = text 
        
        for exp in self.exp_descriptions:
          if exp == text:
            self.exp_descriptions[exp].setObjectName("selected")
          else:
            self.exp_descriptions[exp].setObjectName("not_selected")
          self.exp_descriptions[exp].setStyleSheet(styleSheet)
        self.settings_box.setHidden(False)
        
    
    def study_update(self):
        """Update study type selected by user"""
        rb = self.sender()
        if rb.isChecked():
            self.parent.study_type = rb.text()
        
       

class SecondPage(Page):
    """Class to build the second page (variable selection)."""
    def __init__(self, parent):
        super().__init__(parent=parent, page_number=2)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle('CSV Information')
        self.UIComponents()
        
    def UIComponents(self):
        """Widgets for the page"""
        clearLayout(self.layout)
        # # program title
        self.lbl_title = QLabel('bias.myti.Report', self)
        self.lbl_title.setObjectName("title")
        self.layout.addWidget(self.lbl_title,0,0,1,2)
       
        # # Title Label
        self.lbl_heading = QLabel('Indicate Columns', self)
        self.lbl_heading.setObjectName("heading")
        self.layout.addWidget(self.lbl_heading,1,0,1,2)
        
        column_list = self.get_columns()
        self.selection_defaults = {
          "Positive-associated Subgroup":"Positive-associated",
          "Subgroup":"Subgroup",
          "Metric Name":"Metric",
          "Metric Mean Value":"Mean",
          "Metric Standard Deviation":"Std",
          }
        if self.parent.exp_type == 'Quantitative Misrepresentation':
          self.selection_defaults.update({"Training Prevalence Difference":"Training Prevalence Difference"})
        else:
          self.selection_defaults.update({"Frozen Layers":"Frozen Layers"})

        if self.parent.study_type == 'Compare Bias Mitigation Methods':
          self.selection_defaults.update({"Mitigation Method":"Mitigation"})
        elif self.parent.study_type == 'Study Finite Sample Size Effect':
          self.selection_defaults.update({"Training Data Size":"Data Size"})


        pixmapi = QStyle.StandardPixmap.SP_MessageBoxInformation
        icon = self.style().standardIcon(pixmapi)
        self.selection_labels = {}
        self.information_icon = {}
        self.selection_boxes = {}
        for i, (selection, default) in enumerate(self.selection_defaults.items()):
          self.selection_labels[selection] = QLabel(selection, self)
          self.layout.addWidget(self.selection_labels[selection], i+2, 0, 1, 1)
          self.information_icon[selection] = ClickLabel(self)
          self.information_icon[selection].setPixmap(icon.pixmap(QSize(20, 20)))
          self.information_icon[selection].setObjectName(f"{selection}")
          self.information_icon[selection].clicked.connect(self.add_info)
          self.layout.addWidget(self.information_icon[selection], i+2, 1, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
          self.selection_boxes[selection] = QComboBox(self)
          self.selection_boxes[selection].addItems([default] + column_list)
          self.layout.addWidget(self.selection_boxes[selection], i+2, 2, 1, 2)

        self.addition_info = QLabel()
        self.layout.addWidget(self.addition_info, i+3, 0, 1, 4)
        self.layout.setRowStretch(i+4, 10)
        self.layout.setColumnStretch(2, 10)

    def get_columns(self):
        """Get list of columns from the input csv file"""
        data = pd.read_csv(self.parent.csv_path)
        return list(data.columns)
        
    def check_boxes(self):
      """Update selected variables"""
      for k in self.selection_defaults.keys():
        self.parent.variables[k] = str(self.selection_boxes[k].currentText())
        
    def check_conditions(self):
      """Sanity check to decide if the second page can be appropriately loaded"""
      # # check if the csv file is valid
      if type(self.parent.csv_path) != str or not os.path.exists(self.parent.csv_path): 
        msg = QMessageBox(self) 
        msg.setIcon(QMessageBox.Icon.Warning)  
        msg.setText("Warning: csv file not found!")  
        msg.setWindowTitle("Warning MessageBox") 
        msg.setStandardButtons(QMessageBox.StandardButton.Ok) 
        msg.exec()
        return False
      # # check if amplification type is selected
      elif not self.parent.exp_type in self.parent.experiments:
        msg = QMessageBox(self) 
        msg.setIcon(QMessageBox.Icon.Warning)  
        msg.setText("Warning: please select amplification type!")  
        msg.setWindowTitle("Warning MessageBox") 
        msg.setStandardButtons(QMessageBox.StandardButton.Ok) 
        msg.exec()
        return False
      else:
        return True

    def add_info(self):
      """Update additional information for the variable clicked by user"""
      self.selection_infos = {
          "Positive-associated Subgroup":"The column which indicates the subgroup associated with a more frequent positive outcome label",
          "Subgroup":"The column which indicates subgroup information (e.g., male or female).",
          "Metric Name":"The column which indicates the name of the measured metric.",
          "Metric Mean Value":"The column which includes the measured mean value.",
          "Metric Standard Deviation":"The column which includes the standard deviation of the measurements.",
          "Training Prevalence Difference":"The column which indicates the diease prevalence difference between subgroups during Quantitative Misrepresentation.",
          "Frozen Layers":"The column which includes number of frozen layers during Inductive Transfer Learning.",
          "Mitigation Method":"The column which indicates the type of implemented bias mitigation methods.",
          "Training Data Size":"The column which indicates the value of training data size for finite sample size study.",
          }
      sending_info = self.sender()
      variable = str(sending_info.objectName())
      info_text = self.selection_infos.get(variable)
      self.addition_info.setText(info_text)


        
class FinalPage(Page):
    """Class to build the third page (report display)."""
    def __init__(self, parent, page_number=3):
        super().__init__(parent = parent, page_number=page_number)
        self.setWindowTitle('Myti Results')
        self.current_plot = 0
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.UIComponents()
        
    def run_background(self):
        """Generate figures and descriptions"""
        self.parent.pages["Page 2"].check_boxes()
        self.info_list = result_plotting(self.parent.variables, self.parent.csv_path, self.parent.exp_type, self.parent.study_type)

    def check_conditions(self):
      """Sanity check if the third page can be appropriately loaded"""
      data = pd.read_csv(self.parent.csv_path)
      cols = list(data.columns)
      variables = list(self.parent.variables.values())
      # # check if selected variables existed in the csv file
      if all(item in cols for item in variables):
        return True
      else:
        msg = QMessageBox(self) 
        msg.setIcon(QMessageBox.Icon.Warning)  
        msg.setText("Warning: some variables do not exist in the csv file!")  
        msg.setWindowTitle("Warning MessageBox") 
        msg.setStandardButtons(QMessageBox.StandardButton.Ok) 
        msg.exec()
        return False
        
    def UIComponents(self):
        """Widgets for the page"""
        clearLayout(self.layout)
        # # program title
        self.lbl_title = QLabel('bias.myti.Report', self)
        self.lbl_title.setObjectName('title')
        self.layout.addWidget(self.lbl_title, 0,0,1,2)
        
        # # title for plot selection
        self.lbl_option = QLabel('Plot Options', self)
        self.lbl_option.setObjectName("heading")
        self.layout.addWidget(self.lbl_option, 1,0,1,1)
 
        # # adding example image
        self.example_images = ['../example/example_0.png', '../example/example_1.png']
        self.tile_view = QWidget()
        self.tile_layout = QVBoxLayout()
        self.tile_view.setLayout(self.tile_layout)
        self.tile_figures = []
        self.layout.addWidget(self.tile_view,2,0,2,1)
        
        for ex in self.example_images: 
          self.tile_figures.append(QLabel(self))
          self.tile_figures[-1].setPixmap(QPixmap(ex).scaled(200,150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
          #self.tile_figures[-1].setScaledContents(True)
          self.tile_figures[-1].setObjectName("not_selected")
          self.tile_layout.addWidget(self.tile_figures[-1])
        # event binding # TODO: set in a loop
        self.tile_figures[0].mousePressEvent = lambda x: self.fig_select(figure_number=0)  
        self.tile_figures[1].mousePressEvent = lambda x: self.fig_select(figure_number=1) 
        #self.tile_figures[2].mousePressEvent = lambda x: self.fig_select(figure_number=2) 
        self.tile_layout.addSpacing(70)        
        # Selected Plot
        self.selected_view = QWidget()
        self.selected_layout = QVBoxLayout()
        self.selected_view.setLayout(self.selected_layout)
        self.layout.addWidget(self.selected_view, 2, 1, 2, 2, alignment=Qt.AlignmentFlag.AlignTop)
        self.layout.setColumnStretch(1, 5)
        self.selected_layout.setContentsMargins(0, 0, 0, 0)

        
        # # title for displaying plot
        self.lbl_plot = QLabel('Selected Plot', self)
        self.lbl_plot.setObjectName("heading")
        self.layout.addWidget(self.lbl_plot,1,1,1,1)
        # # position for selected plot
        self.lbl_selected_plot = QLabel('Please click one of the plots on the left to display here.', self)
        #self.lbl_selected_plot.resize(400,300)
        self.selected_layout.addWidget(self.lbl_selected_plot, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.selected_layout.setStretch(0,5)
        # # description for the plot
        self.lbl_selected_dscp = QLabel(self)
        self.lbl_selected_dscp.resize(400,100)
        self.lbl_selected_dscp.setWordWrap(True) 
        self.selected_layout.addWidget(self.lbl_selected_dscp)
        #self.selected_layout.addSpacing(1)
        # references
        self.lbl_references = QLabel("For additional reading: \n Y. Zhang, A. Burgon, N. Petrick, B. Sahiner, G. Pennello, R. K. Samala*, “Evaluation of AI bias mitigation algorithms by systematically promoting sources of bias”, RSNA Program Book (2023).\nA. Burgon, Y. Zhang, B. Sahiner, N. Petrick, K. H. Cha, R. K. Samala*, “Manipulation of sources of bias in AI device development”, Proc. of SPIE (2024).")
        self.lbl_references.setObjectName("references")
        self.lbl_references.setWordWrap(True)
        self.layout.addWidget(self.lbl_references, 4, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignBottom)
        # # button to save figure
        self.btn_save_fig = QPushButton('Save Report', self)
        self.btn_save_fig.resize(self.btn_save_fig.sizeHint())
        self.btn_save_fig.clicked.connect(self.save_fig)
        self.layout.addWidget(self.btn_save_fig, 0, 2, 1, 1)
        
        self.layout.setRowStretch(3, 10)

    def fig_select(self, figure_number, event=None):
        """Update the selected figure by user"""
        for i, w in enumerate(self.tile_figures):
          if i == figure_number:
            w.setObjectName("selected")
          else:
            w.setObjectName("not_selected")
          w.setStyleSheet(styleSheet)
        
        # Set the large image and descriptiong
        self.lbl_selected_plot.setPixmap(QPixmap(self.example_images[figure_number]).scaled(440,330, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        info = self.info_list[figure_number]
        self.lbl_selected_dscp.setText(info)
        self.current_plot = figure_number

    def quit_page(self):
        shutil.rmtree('../example/tmp/')
        widget.close()
    
    def save_fig(self):
        """Save the figure and description"""
        name = QFileDialog.getSaveFileName(self, 'Save File', 'saved_report.png', "Images (*.png *.jpg);;PDF files (*.pdf)")
        save_report(self.info_list[self.current_plot], self.example_images[self.current_plot], name[0])

class MainWindow(QMainWindow):
    """Class for the main window including pages, logo, navigation bars."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("bias.myti.Report")
        self.setObjectName("main window")
        self.setWindowIcon(QIcon("UI_assets/fda_logo.jpg"))
        self.setGeometry(100, 100, 1200, 700) 
        
        # Set argument placeholders
        self.csv_path = '../example/example.csv'
        self.exp_type = 'Quantitative Misrepresentation'
        self.study_type = 'None'
        self.current_page = 1
        self.variables = {}
        
        # Constants
        self.experiments = {"Quantitative Misrepresentation":'The Quantitative Misrepresentation approach applies data selection prior to training so that the disease prevalence is different for different patient subgroups.\n' + \
              '\nThe degree to which bias is promoted can be controlled by changing the degree of prevalence gap between subgroups.',
              "Inductive Transfer Learning":'The Inductive Transfer Learning approach applies an extra step of transfer learning prior to target model fine-tuning to promote spurious correlation between model task and subgroup infomation.\n' + \
            '\nThe degree to which bias is promoted can be controlled by number of frozen layers in last step fine-tuning.'}
        
        
        # Load
        self.load_GUI()
        self.show()
        
        self.change_page(1)
    
    def load_GUI(self):
        """Widgets for the main window"""
        # Set up overall layout
        self.main_layout = QGridLayout()
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        
        # Create sidebar
        self.make_sidebar()
        self.main_layout.addWidget(self.sidebar, 0, 0, 2, 1)
        
        # Make Tab widget (to hold pages)
        self.tab_widget = QTabWidget()
        self.set_pages()
        self.main_layout.addWidget(self.tab_widget, 0, 1, 1, 1)
        
        # Create Navigation bar
        self.make_navbar()
        self.main_layout.addWidget(self.navigation_bar,1, 1, 1, 1)
        
        # Format main layout
        self.main_layout.setColumnStretch(0, 40)
        self.main_layout.setColumnStretch(1, 200)
        self.main_layout.setRowStretch(0,100)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)
    
    
    def set_pages(self):
        """Set up three pages"""
        self.pages = {}
        self.page_layouts = {}
        self.page_classes = {"Page 1":InitialPage, "Page 2":SecondPage, "Page 3": FinalPage}
        for p in range(1,4):
          self.pages[f"Page {p}"] = self.page_classes[f"Page {p}"](self)
          self.tab_widget.addTab(self.pages[f"Page {p}"], "")
        
        self.sidebar_buttons["Page 1"].clicked.connect(lambda x: self.change_page(1))
        self.sidebar_buttons["Page 2"].clicked.connect(lambda x: self.change_page(2))
        self.sidebar_buttons["Page 3"].clicked.connect(lambda x: self.change_page(3))
    
    def change_page(self, page_number:int, *args, **kwargs):
        """Naviagte between three pages"""
        if not self.pages[f"Page {page_number}"].check_conditions():
          return
        self.tab_widget.setCurrentIndex(page_number-1)
        self.pages[f"Page {page_number}"].load()
        
        self.current_page = page_number
        self.update_navbar()
        
        for id, w in self.sidebar_buttons.items():
          if id == f"Page {page_number}":
            w.setObjectName("active")
            w.setEnabled(True)
          else:
            w.setObjectName("inactive")
          w.setStyleSheet(styleSheet)
      
    def next_page(self):
        self.change_page(self.current_page + 1)
    
    def prev_page(self):
        self.change_page(self.current_page - 1)
    
    def make_sidebar(self):
        """Add side bars"""
        self.sidebar_buttons = {}
        self.sidebar_layout = QVBoxLayout()
        
        self.sidebar_layout.addStretch(1)
        self.fda_logo = QLabel()
        self.fda_logo.setObjectName("fda_logo")
        self.fda_logo.setPixmap(QPixmap("UI_assets/fda_logo_full.jpg").scaled(200,150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.sidebar_layout.addWidget(self.fda_logo)
        self.sidebar_layout.addStretch(1)
        
        page_icons = ["UI_assets/file-line-icon.svg", "UI_assets/sliders-icon.svg", "UI_assets/graph-icon.svg"]
        page_names = ["Input", "Variables", "Report"]
        
        for i, icon_path in enumerate(page_icons):
          p = i + 1
          self.sidebar_buttons[f"Page {p}"] = QPushButton(page_names[i], self)
          self.sidebar_buttons[f"Page {p}"].setIcon(QIcon(icon_path))
          self.sidebar_layout.addWidget(self.sidebar_buttons[f"Page {p}"])
          self.sidebar_buttons[f"Page {p}"].setEnabled(False)
          self.sidebar_buttons[f"Page {p}"].setObjectName("inactive")
          
        self.sidebar_buttons["Page 1"].setEnabled(True)
        self.sidebar_buttons["Page 1"].setObjectName("active")
          
        self.sidebar_layout.addStretch(10)
        self.sidebar_layout.setSpacing(0)
        self.sidebar_layout.setContentsMargins(0,0,0,0)
        
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setLayout(self.sidebar_layout)
    
    def make_navbar(self):
        """Add naviagtion buttons"""
        self.nav_layout = QHBoxLayout()
        
        self.OSEL_label = QLabel("OSEL")
        self.OSEL_label.setObjectName("osel_label")
        self.nav_layout.addWidget(self.OSEL_label)
        self.OSEL_desc = QLabel("Accelerating patient access to innovative, safe, and effective medical devices through best-in-the-world regulatory science")
        self.OSEL_desc.setObjectName("osel_desc")
        self.nav_layout.addWidget(self.OSEL_desc)
        self.nav_layout.addStretch(10)
        
        self.prev_button = QPushButton("Previous Page")
        self.prev_button.setObjectName("navigation_button")
        self.prev_button.clicked.connect(self.prev_page)
        self.nav_layout.addWidget(self.prev_button)
          
        self.next_button = QPushButton("Next Page")
        self.next_button.setObjectName("navigation_button")
        self.next_button.clicked.connect(self.next_page)
        self.nav_layout.addWidget(self.next_button)      
          
        self.navigation_bar = QWidget()
        self.navigation_bar.setObjectName("navigation_bar")
        self.navigation_bar.setLayout(self.nav_layout)
        
        self.update_navbar()
    
    def update_navbar(self):
        if self.current_page == 1:
          self.prev_button.hide()
        else:
          self.prev_button.show()
        
        if self.current_page == len(self.pages):
          self.next_button.hide()
        else:
          self.next_button.show()
  
if __name__ == '__main__':
    app = QApplication(sys.argv)
    styleSheet = Path('UI_assets/style_sheet.qss').read_text()
    app.setStyleSheet(styleSheet)
    window = MainWindow()    
    sys.exit(app.exec())
