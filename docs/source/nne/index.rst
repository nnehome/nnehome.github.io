.. _home:
.. _nne:

NNE (Limited-information)
=========================

|

.. _top:

We provide an overview and code for the Neural Net Estimator (NNE) (based on `"Estimating Parameters of Structural Models Using Neural Nets," Wei and Jiang 2025 <https://pubsonline.informs.org/doi/10.1287/mksc.2022.0360>`_).

|

Overview
---------------

NNE is an approach to estimate structural econometric models (e.g., discrete choice, consumer search, games). Suppose that the structural model's parameter vector is :math:`\boldsymbol{\theta}`. The basic idea is to train a neural net that can recognize the value of :math:`\boldsymbol{\theta}` from a vector of data moments. These data moments are specified by us researchers. The training examples for the neural net come from simulating datasets using the structural model. Specifically,

#. Draw a value of :math:`\boldsymbol{\theta}` from a prior. Given this value and the real-data product/consumer attributes, use the structural model to simulate a dataset. Compute the specified moment vector for this dataset.

#. Repeat the above step :math:`L` times to obtain :math:`L` pairs of values of :math:`\boldsymbol{\theta}` and moment vectors. These pairs are our training examples.

#. Use the :math:`L` examples to train a neural net that predicts the value of :math:`\boldsymbol{\theta}` from moment vector.

Finally, we plug the real-data moment vector into the trained neural net to obtain an estimate of :math:`\boldsymbol{\theta}`. We see that NNE does not require evaluating likelihood or moment function. Thus, it is particularly useful for estimating structural models where: (i) likelihood/moment function has no closed forms and is difficult to simulate accurately, or (ii) the simulated likelihood/moment function is difficult to optimize.

In the paper, it is shown that as we increase :math:`L`, the neural net converges to the Bayesian posterior mean of :math:`\boldsymbol{\theta}` given the specified moments (i.e., a limited-information posterior). The paper also shows how we can train the neural net to estimate the posterior variance.

This NNE is most useful when researchers have clear intuition about what data moments could identify the structural model. When this is not the case, one may want to use the `full-information NNE <https://nnehome.github.com/fnne/index.html>`_.

|

Applications
------------

We provide Matlab code for two examples:

* **An AR1 model**. This is a toy example to illustrate how NNE works.

* **A consumer search model**. This examples shows the computational and accuracy advantages of NNE.

You can find the code at this `GitHub directory <https://github.com/nnehome/nne-matlab-code>`__, and code documentation at the :ref:`AR1 model <nne_ar1>` page and the :ref:`search model <nne_search>` page.

|

.. toctree::
   :hidden:

   Overview <self>
   Search Model <nne_search>
   AR1 Model <nne_ar1>
