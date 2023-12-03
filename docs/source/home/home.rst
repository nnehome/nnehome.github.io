:html_theme.sidebar_secondary.remove:

:parenttoc: True

.. _home:

Welcome to NNE
===============

|

.. _top:

This website provides a guide and code of the neural net estimator (NNE) (`paper`_). NNE exploits machine learning techniques to estimate existing econometric models. It can achieve good estimation accuracy at light computational costs, and thus provides an alternative to simulated maximum likelihood or simulated method of moments. 

Below, we describe an `overview`_ of NNE, its `step-by-step procedure`_, and the `applicability`_ to marketing/economics problems.

We also provide Matlab code for two applications of NNE. You're welcome to modify the code to estimate other econometric models. You can find the code at this `GitHub <https://github.com/nnehome/nne-matlab>`_ repository. Please also find the documentation for the code at the :ref:`consumer search <code_1>` page and the :ref:`AR1 model <code_2>` page.

|

Overview
---------------

A (structural) econometric model specifies some outcome of interest :math:`\boldsymbol{y}\equiv\{y_i\}_{i=1}^{n}` as a function of some observed attributes :math:`\boldsymbol{x}\equiv\{\boldsymbol{x}_i\}_{i=1}^{n}`  and some unobserved attributes :math:`\boldsymbol{\epsilon}`. The function is often an economic model such as random utility maximization, sequential search, game, etc. The outcome of interest can be consumer choice, sales, etc. The observed attributes often are consumer and product characteristics.

So we can denote a structural econometric model as :math:`\boldsymbol{y} = \boldsymbol{q}(\boldsymbol{x}, \boldsymbol{\epsilon},  \boldsymbol{\theta})`, where :math:`\boldsymbol{\theta}` is the parameter of the model. The task of structural estimation can be described as recovering the parameter :math:`\boldsymbol{\theta}` from data :math:`\{\boldsymbol{x, y}\}`.

The basic idea of NNE is to train neural nets to recognize :math:`\boldsymbol{\theta}` from data :math:`\{\boldsymbol{x, y}\}`. 

To train such a neural net, we only require it is possible to simulate :math:`\boldsymbol{y}` using the econometric model. Specifically, we first draw many different values of the parameter. Given each parameter value, use the structural model to generate a copy of data. Then, these many parameter values and their corresponding copies of data become the training examples for the neural net. The step-by-step procedure below gives more details.
  
|

Step-by-step procedure
-----------------------

Below we list the usual procedure of NNE. We use :math:`\ell` to index the training examples that we use to train the neural net.

#. **Simulate data.** For each :math:`\ell`, draw parameter values :math:`\boldsymbol{\theta}^{(\ell)}` from a parameter space :math:`\Theta`. Use the structural econometric model to simulate a set of outcomes :math:`\boldsymbol{y}^{(\ell)}`, under the parameter :math:`\boldsymbol{\theta}^{(\ell)}` and the observed attributes :math:`\boldsymbol{x}`.

#. **Summarize data.** For each :math:`\ell`, summarize the data :math:`\{\boldsymbol{y}^{(\ell)}, \boldsymbol{x}\}` into a set of data moments :math:`\boldsymbol{m}^{(\ell)}`. 

#. **Train a neural net.** Repeat step 1-3 for :math:`\ell=1,...,L` to construct the training examples :math:`\{\boldsymbol{m}^{(\ell)},\boldsymbol{\theta}^{(\ell)}\}_{\ell=1}^{L}`. We can also repeat these steps more times to create validation examples. Use these examples to train a neural net. 

#. **Get the estimate.** Plug the real data moments into the neural net to obtain an estimate of :math:`\boldsymbol{\theta}`.

Some practical notes:

* We specify :math:`\Theta` so that it likely contains the true :math:`\boldsymbol{\theta}`. If we have a prior, we may also draw :math:`\boldsymbol{\theta}^{(\ell)}` from the prior distribution.
 
* We specify :math:`\boldsymbol{m}` so that it contains relevant information for recovering :math:`\boldsymbol{\theta}`. Common examples include the mean of :math:`\boldsymbol{y}` and the covariances between :math:`\boldsymbol{y}` and :math:`\boldsymbol{x}`. It is generally OK to include possibly irrelevant or redundant moments in :math:`\boldsymbol{m}` -- the performance of NNE is relatively robust to redundant moments.
 
* We can use mean-square-error loss to train NNE. Other loss functions can train the neural net to give measures of statistical accuracy in addition to point estimates. See our `paper`_ for details.

|

Applicability
---------------

Increasing complexity of models in economics/marketing means there is often no closed-form expressions of likelihood or moment functions. So reseachers often rely on simulation-based estimators such as simulated maximum likelihood (SMLE) or simulated method of moments (SMM). NNE is a simulation-based estimator as well. But NNE offers sizable speed and accuracy gains over SMLE/SMM in some applications, making estimation much more tractable. One particular application in marketing that benefits from NNE is consumer sequential search. We have studied it extensively in our `paper`_. You can find our code on the :ref:`consumer search <code_1>` page.

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
     - It does not require computing integrals over the unobservables (:math:`\boldsymbol{\epsilon}`) in the structural econometric model. It only requires being able to simulate data using the econometric model.
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
     - Closed-form expressions are available for likelihood/moments. The main estimation burden comes from other than the simulations to evaluate likelihood/moments.
   * - **Examples**: discrete choices with rich unobserved heterogeneity, sequential search, choices on networks.
     - **Examples**: dynamic choice or games where the main burden is solving policy functions.

|

|

Paper
---------------

|

Yanhao 'Max' Wei and Zhenling Jiang (2023). "Estimating Parameters of Structural Models with Neural Networks." 

`SSRN link <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3496098#>`_

|

|

|

|

