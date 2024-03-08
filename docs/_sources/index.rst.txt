.. bias.myti.report documentation master file, created by
   sphinx-quickstart on Thu Jan 11 12:34:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bias.myti.report
================
Bias.myti.report is a bias visualization tool designed to facilitate the systematic comparison of bias mitigation methods through the creation of multiple models with a range of AI biases. The tool provides the guidance on how to implement two approaches, namely quantitative misrepresentation and inductive transfer learning, for such bias amplification. Additional controls over the degree to which bias is amplified can be taken in both approaches. The tool can be readily incorporated into the AI development pipeline so that the bias mitigation methods can be evaluated under specific user cases.



Getting Started
===============

Bias Amplification
------------------

To apply bias amplification, a data set with both class and subgroup attribute information is required. Please follow steps below for implementation:

**1. Quantitative Misrepresentation**

**Step 1.** Partition your own data set into training, validation and test sets.

**Step 2.** Create a list of new training sets with a range of disease prevalence difference between subgroups.

**Step 3.** Train your customized models with sampled training sets from **Step 1**.

**Step 4.** Deploy your models on the test set.

**Step 5.** Calculate metrics to measure subgroup bias (subgroup predicted prevalence, sensitivity, etc.)

**2. Inductive Transfer Learning**

**Step 1.** Partition your own data set into training, validation and test sets.

**Step 2.** Train your customized model to classify subgroup attributes.

**Step 3.** Fine-tune the result model from **Step 2** with different number of freezing layers for target task classification.

**Step 4.** Deploy your final models on the test set.

**Step 5.** Calculate metrics to measure subgroup bias (subgroup predicted prevalence, sensitivity, etc.)

Demonstrations for both approaches can be found in two separate example Jupyter notebooks included in the `examples folder`_. 

Bias Visualization
------------------

To visualize bias using bias.myti.report GUI tool, an **input csv** which contains results from bias amplification (and mitigation) experiments is required. To make a comparison between bias mitigation methods, users have to implement mitigation algorithms by their choice and aggregate the results into one single csv file.



.. _examples folder: https://github.com/DIDSR/myti.report/tree/main/example


Terminology
===========
* ``model bias``: A systematic difference in performance between subgroups.
* ``bias amplification``: A process to increase the performance difference between subgroups.
* ``quantitative misrepresentation``: A bias amplification approach that applies a data selection process in the training set so that the disease prevalence between subgroups are different.
* ``inductive transfer learning``: A bias amplification approach that applies a two-step transfer learning scheme, where AI model is trained to classify subgroup attributes during the first step. In the second step, the model is fine-tuned to perform clinical tasks.

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
