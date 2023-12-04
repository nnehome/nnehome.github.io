:html_theme.sidebar_secondary.remove:

.. pNNE documentation master file, created by
   sphinx-quickstart on Mon Sep 18 12:08:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 3

.. toctree::
   :hidden:

    Home <home/home>
    Consumer Search <code_consumer_search/code_consumer_search>
    AR1 Model <code_ar1_model/code_ar1_model>


Welcome to NNE
===============

|

.. _top:

This website provides a guide for and the code of the neural net estimator (NNE) (Wei and Jiang 2023, 
`SSRN link <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3496098#>`_). 
NNE exploits machine learning techniques to estimate existing econometric models. 
It is a simulation-based estimator and provides an alternative to simulated maximum likelihood or simulated method of moments. 
It offers sizable computational and accuracy gains in suitable applications.

Below, we describe an `overview`_ of NNE, its `step-by-step procedure`_, and the `applicability`_ to marketing/economics problems.

We also provide Matlab code for two applications of NNE: a consumer search model and an AR1 model. 
The AR1 is a good example to illustrate the concept of NNE, whereas the consumer search application demonstrates computational and accuracy advantages of NNE. 
You can find the code at this `GitHub <https://github.com/nnehome/nne-matlab>`_ repository.
Please also find the code documentation at the :ref:`consumer search <code_consumer_search>` page and the :ref:`AR1 model <code_ar1_model>` page. 
You're welcome to modify the code to estimate other econometric models. 

|

Overview
---------------

A (structural) econometric model specifies some outcome of interest :math:`\boldsymbol{y}\equiv\{y_i\}_{i=1}^{n}` as a function :math:`\boldsymbol{q}` of some observed attributes :math:`\boldsymbol{x}\equiv\{\boldsymbol{x}_i\}_{i=1}^{n}`  and some unobserved attributes :math:`\boldsymbol{\varepsilon}`. The function :math:`\boldsymbol{q}` is often an economic model such as random utility maximization, sequential search, game, etc. The outcome of interest :math:`\boldsymbol{y}` can be consumer choice, product sales, etc. The observed attributes :math:`\boldsymbol{x}` are often consumer and product characteristics.

So we can denote a structural econometric model as :math:`\boldsymbol{y} = \boldsymbol{q}(\boldsymbol{x}, \boldsymbol{\epsilon},  \boldsymbol{\theta})`, where :math:`\boldsymbol{\theta}` is the parameter vector of the model. The task of structural estimation can be described as recovering the parameter :math:`\boldsymbol{\theta}` from data :math:`\{\boldsymbol{x, y}\}`.

The basic idea of NNE is to train neural nets to recognize :math:`\boldsymbol{\theta}` from data :math:`\{\boldsymbol{x, y}\}`. 

To train such a neural net, we only require being able to simulate :math:`\boldsymbol{y}` using the econometric model. Specifically, we first draw many different values of the parameter. Given each parameter value, use the structural model to generate a copy of data. Then, these many parameter values and their corresponding copies of data become the training examples for the neural net. The step-by-step procedure below gives more details.
  
|

Step-by-step procedure
-----------------------

Below we list the usual procedure of NNE. We use :math:`\ell` to index the training examples that we use to train the neural net.

#. **Simulate data.** For each :math:`\ell`, draw parameter vector :math:`\boldsymbol{\theta}^{(\ell)}` from a parameter space :math:`\Theta`. Given this parameter :math:`\boldsymbol{\theta}^{(\ell)}` and the observed attributes :math:`\boldsymbol{x}`, use the structural econometric model to simulate outcomes :math:`y_i^{(\ell)}` for :math:`i=1,...,n`. Let :math:`\boldsymbol{y}^{(\ell)}\equiv\{y_i^{(\ell)}\}_{i=1}^{n}`.

#. **Summarize data.** For each :math:`\ell`, summarize the data :math:`\{\boldsymbol{y}^{(\ell)}, \boldsymbol{x}\}` into a set of data moments :math:`\boldsymbol{m}^{(\ell)}`. 

#. **Train a neural net.** Repeat steps 1-3 for :math:`\ell=1,...,L` to construct the training examples :math:`\{\boldsymbol{m}^{(\ell)},\boldsymbol{\theta}^{(\ell)}\}_{\ell=1}^{L}`. We can also repeat steps 1-3 more times to create validation examples. Use these examples to train a neural net. 

#. **Get the estimate.** Plug the real data moments into the neural net to obtain an estimate of :math:`\boldsymbol{\theta}`.

Some practical notes:

* We specify :math:`\Theta` so that it likely contains the true :math:`\boldsymbol{\theta}`. If we have a prior, we may also draw :math:`\boldsymbol{\theta}^{(\ell)}` from the prior distribution.
 
* We specify :math:`\boldsymbol{m}` so that it contains relevant information for recovering :math:`\boldsymbol{\theta}`. Common examples include the mean of :math:`\boldsymbol{y}` and the covariances between :math:`\boldsymbol{y}` and :math:`\boldsymbol{x}`. It is generally OK to include possibly irrelevant or redundant moments in :math:`\boldsymbol{m}` -- the performance of NNE is relatively robust to redundant moments.
 
* We can use mean-square-error loss to train NNE. Other loss functions can train the neural net to give measures of statistical accuracy in addition to point estimates. See our paper referenced :ref:`above <top>` for details.

|

Applicability
---------------

The increasing complexity of models in economics/marketing means there are often no closed-form expressions of likelihood or moment functions. 
So reseachers thus rely on simulation-based estimators such as simulated maximum likelihood (SMLE) or simulated method of moments (SMM). 
NNE is a simulation-based estimator as well. But NNE offers sizable speed and accuracy gains over SMLE/SMM in some applications, 
making estimation much more tractable. One particular application in marketing that benefits from NNE is consumer 
sequential search. We have studied it extensively in our paper referenced :ref:`above <top>`. 
You can find our code on the :ref:`consumer search <code_consumer_search>` page.

The table below summarizes the main properties of NNE as well as its suitable applications.

|

.. _main-properties-table:

.. list-table:: 
   :widths: 10 90
   :header-rows: 1
   :class: table-header-centered

   * - 
     - Main Properties
   * - 1
     - It does not require computing integrals over the unobservables (:math:`\boldsymbol{\varepsilon}`) in the structural econometric model. It only requires being able to simulate data using the econometric model.
   * - 2
     - It does not require optimizing an (potentially non-smooth) objective function as in extremum estimators (e.g., SMLE, SMM, indirect inference).
   * - 3
     - It is more robust to redundant moments when compared to SMM/GMM.
   * - 4
     - It computes a measure of statistical accuracy as a byproduct.

.. list-table:: 
   :widths: 50 50
   :header-rows: 1
   :class: table-header-centered

   * - Suitable Applications
     - Less Suitable Applications
   * - A large number of simulations are needed to evaluate likelihood/moments. The SMLE/SMM objective is difficult to optimize. There lacks clear guidance on moment choice. Formulas of standard errors are not yet established.
     - Closed-form expressions are available for likelihood/moments. The main estimation burden comes from sources other than the simulations to evaluate likelihood/moments.
   * - **Examples**: discrete choices with rich unobserved heterogeneity, sequential search, choices on networks.
     - **Examples**: dynamic choice or games where the main burden is solving policy functions.

|

|


