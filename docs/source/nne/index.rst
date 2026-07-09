.. _home:
.. _nne:

NNE (Limited-information)
=========================

|

.. _top:

We provide an overview and code for the Neural Net Estimator (NNE), based on the paper :note-text:`"Estimating Parameters of Structural Models Using Neural Nets."`


1. Overview
---------------

NNE is an approach to estimate structural econometric models (e.g., discrete choice, consumer search, games). Suppose that the structural model's parameter vector is :math:`\boldsymbol{\theta}`. The basic idea is to train a neural net that can recognize the value of :math:`\boldsymbol{\theta}` from a vector of data moments (or more broadly, data statistics). These moments are specified by researchers. The training examples for the neural net come from datasets simulated using the structural model. Specifically,

#. Draw a value of :math:`\boldsymbol{\theta}` from a prior. Given this value and the real-data product/consumer attributes, use the structural model to simulate a dataset. Compute the specified moment vector for this dataset.

#. Repeat the above step :math:`L` times to obtain :math:`L` pairs of values of :math:`\boldsymbol{\theta}` and moment vectors. These pairs are our training examples.

#. Use the :math:`L` examples to train a neural net that predicts the value of :math:`\boldsymbol{\theta}` from moment vector.

Finally, we plug the real-data moment vector into the trained neural net to obtain an estimate of :math:`\boldsymbol{\theta}`. Thus, NNE does not require evaluating likelihood or moment function. Consequently, it is particularly useful for estimating structural models where: (i) likelihood/moment function has no closed form and is difficult to simulate accurately, or (ii) the simulated likelihood/moment function is difficult to optimize. Further, this NNE does not require i.i.d. data --- it works for time-series, network data, etc.

In the paper, it is shown that as we increase :math:`L`, the neural net converges to the Bayesian posterior mean of :math:`\boldsymbol{\theta}` given the specified moments (i.e., a limited-information posterior). The paper also shows how we can train the neural net to estimate the posterior variance.

This NNE is most useful when we know a set of candidate moments that could identify :math:`\boldsymbol{\theta}`. When this is not the case, one should try `full-information NNE <https://nnehome.github.io/fnne/index.html>`_. However, full-info NNE requires i.i.d. structure in data.


2. Applications
---------------

We provide Matlab code for two examples:

* **An AR1 model**. This is a toy example to illustrate how NNE works.

* **A consumer search model**. This example shows the computational and accuracy advantages of NNE.

You can find the code at this `GitHub directory <https://github.com/nnehome/nne-matlab-code>`__, and code documentation at the :ref:`AR1 model <nne_ar1>` page and the :ref:`search model <nne_search>` page.

|

.. toctree::
   :hidden:

   Overview <self>
   Search Model <nne_search>
   AR1 Model <nne_ar1>
