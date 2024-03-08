.. bias.myti.Report documentation master file, created by
   sphinx-quickstart on Thu Jan 11 12:34:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bias.myti.Report
================
Bias.myti.Report aims to increase understanding of  AI model bias as well as the effectiveness of implemented mitigation methods. The tool provides the guidance on how to systematically amplify model bias and help the user to visualize such bias. Two approaches, including quantitative misrepresentation and inductive transfer learning, can be implemented to for bias amplification. Additional controls over the degree to which bias is amplified can be taken in both approaches. Such approaches facilitate a systematic comparison of user-implemented bias mitigation methods.



Getting Started
===============
* Bias Amplification Implementation

To implement proposed bias amplification approach, please follow the instructions in example Jupyter notebooks included in the `examples folder`_. The user can either uses the MIDRC Open-A1 data set provided in the example or their own data set. Please note that the data set must have subgroup attribute information to amplify and measure bias. 

* Bias Visualization

To visualize bias using myti.report GUI tool, an **input csv** which contains results from bias amplification (and mitigation) experiments. To make comparison between bias mitigation methods, the user has to implement the mitigation algorithm by themselves and aggregate the results into a single csv file.



.. _examples folder: https://github.com/DIDSR/myti.report/tree/main/example


Terminology
===========
* ``model bias``: A systematic difference in performance between subgroups.
* ``bias amplification``: A process to increase the performance difference between subgroups.
* ``quantitative misrepresentation``: A bias amplification approach that apply data selection in the training set so that the disease prevalence between subgroups are different.
* ``inductive transfer learning``: A bias amplification approach that applied a two-step transfer learning scheme, where AI model is train to classify subgroup attributes during the first step. In the second step, the model is fine-tuned to perform clinical tasks.

Function and Class Documentation
================================
.. toctree::
   :maxdepth: 2

   self
   src


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
