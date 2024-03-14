bias.myti.report
================

Bias.myti.report facilitates the systematic comparison of bias mitigation methods through the creation of multiple models with different degrees of bias. Bias.myti.report has two main components:

1. `Implementation instructions for two bias amplification approaches`_ (quantitative misrepresentation and inductive transfer learning)
2. A `visualization GUI`_ to assist with the interpretation of results.


Getting Started
===============

This repository contains two example notebooks in the `examples folder`_ that demonstrate how to amplify bias using data from the MIDRC open-A1 repository. Additionally, the `examples folder`_ contains several example csv files, containing example inputs in to the visualization GUI. To implement the amplification approaches using your own data and/or models, see the implementation instructions in the next section. 


.. _Implementation instructions for two bias amplification approaches:

Bias Amplification
==================

Each of the bias amplification approaches outlined in this section amplify bias by promoting the use of subgroup information during classification through the development of a "learning shortcut". Both of these approaches are described in terms of the amplification of bias between two distinct subgroups for a binary classification model, and can amplify bias to varying degrees. 

Quantitative Misrepresentation
------------------------------

This approach amplifies bias by manipulating the disease prevalence between subgroups in the *training* data. An example implementation is included in the `examples folder`_. To apply this approach to your own model and/or data, follow the steps below:

1. Partition your data into training, validation and test sets, each equally stratified by both subgroup and class.
2. Sample the training data into a series of new training sets with varying subgroup disease prevalence. The number of patients within each subgroup should always remain equal and the overall disease prevalence fixed at 50%, however the prevalence within each subgroup will differ. For example, if training a model to classify patients as positive or negative and amplifying the bias between subgroups A and B, one sampled training set may have the following distributions:

  +----------+------------+------------+-------+
  |          | Subgroup A | Subgroup B | Total |
  +----------+------------+------------+-------+
  | Positive | 10%        | 40%        | 50%   |
  +----------+------------+------------+-------+
  | Negative | 40%        | 10%        | 50%   |
  +----------+------------+------------+-------+
  | Total    | 50%        | 50%        | 100%  |
  +----------+------------+------------+-------+
  
3. Train a collection of models using the training sets with different prevalences as sampled in step 2.
4. Deploy your models on the equally stratified test set from step 1.
5. Calculated performance metrics (e.g., sensitivity, AUROC) for each subgroup as well as the test set overall.
6. Load the calculated performance values into the `included GUI`_.

Inductive Transfer Learning
---------------------------

This approach amplifies bias by training the model in two steps. The model learns the image characteristics associated with subgroup in the first, and is forced to use them to determine class in the second. An example implementation is included in the `examples folder`_. To apply this approach to your own model and/or data, follow the steps below:

1. Partition your data into training, validation and test sets, each equally stratified by both subgroup and class.
2. Without freezing any of the model's layers, train the model to classify sample **subgroup**.
3. Using the trained model created in step 2, finetune the model to classify sample class. Varying degrees of bias can be created by freezing different numbers of layers in this step; bias amplification increases with more layers frozen.
4. Deploy your models on the equally stratified test set from step 1.
5. Calculated performance metrics (e.g., sensitivity, AUROC) for each subgroup as well as the test set overall.
6. Load the calculated performance values into the `included GUI`_.


.. _visualization GUI:

Bias Visualization
==================

This repository includes a GUI created to assist with the visualization and interpretation of the results created using quantitative misrepresentation and inductive transfer learning. To make a comparison between bias mitigation methods, users have to implement mitigation algorithms by their choice and aggregate the results into one single csv file, an example of the expected file structure is included in the `examples folder`_.


.. _examples folder: https://github.com/DIDSR/bias.myti.report/tree/main/example
.. _included GUI: https://github.com/DIDSR/bias.myti.report/tree/main/src/mytiGUI.py

Terminology
===========

**Class**
	The sample characteristic by which the model is classifying samples. For example, in the example notebooks included in this repository, the model is tasked with determining patient COVID status, so the two potential sample classes are COVID positive and COVID negative.

**Subgroup**
	A subset of samples grouped by a shared characteristic not associated with the classification task. For example, in the example notebooks included in this repository, the samples are divided into subgroups based on patient sex, an attribute which is not associated with the classification task of COVID status.

**Model bias**
	A systematic difference in performance between subgroups.
 
**Bias amplification**
  A process which deliberately increases the model bias between specified subgroups.
  
**Learning shortcut**
  The usage of characteristics characteristics unrelated to the sample's class during classification. These characteristics are often common amoungst samples within a specific class in the development data, but not throughout the entire intended population, resulting in decreased performance or model bias when the model is deployed on a larger population.

**Quantitative misrepresentation**
  A bias amplification approach that applies a data selection process to the selection of model development data sets such that the disease prevalence between subgroups is different, which promotes the development of learning shortcuts.

**Inductive transfer learning**
  A bias amplification approach that applies a two-step transfer learning process. The AI model is first trained to classify samples by subgroup, then finetuned to classify samples by class. This process promotes the usage of subgroup-related characteristics during class classification, promoting the development of learning shortcuts.

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
