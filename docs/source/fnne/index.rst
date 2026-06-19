.. _fnne:

Full-information NNE
====================

|

We provide an overview and code for the full-information Neural Net Estimator (full-info NNE), based on `"Estimating and Assessing Identification of Structural Models via Deep Learning," Wei and Jiang 2026 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6774178>`__. 

|

Overview
--------

NNE is an approach to estimate structural econometric models (e.g., discrete choice, consumer search, games). Let the structural model's parameter vector be :math:`\boldsymbol{\theta}`. The basic idea is to train a neural net that can recognize the value of :math:`\boldsymbol{\theta}` from data. The :ref:`original NNE <nne>` uses researcher-specified moments as input to the neural net. In contrast, full-info NNE frees researchers from moment specification and *uses the entire dataset as input*.

The key challenge is that with a large input like an entire dataset, we typically must put a structure on the neural net architecture to make training feasible. Meanwhile, we want the architecture to be non-restrictive such that the neural net can still use all the information in data. It turns out that, assuming i.i.d. data (e.g., cross-sectional or panel), we can use an architecture  exploiting the i.i.d. structure to achieve these two goals.

The full-information NNE is trained as follows.

#. Draw a value of :math:`\boldsymbol{\theta}` from a prior. Given this value and the real-data product/consumer attributes, use the structural model to simulate a dataset.

#. Repeat the above step :math:`L` times to obtain :math:`L` pairs of values of :math:`\boldsymbol{\theta}` and datasets. These pairs are our training examples.

#. Use the :math:`L` examples to train a neural net with the "two-part architecture" to predict the value of :math:`\boldsymbol{\theta}` from a dataset.

The "two-part architecture" first transforms each observation into a vector of features, and then uses the average features across observations to predict the value of :math:`\boldsymbol{\theta}`. The architecture is easy to implement because it can be conveniently coded as a convolutional neural net (CNN). In the paper, it is shown that this architecture is not restrictive, in the sense that the neural net converges to the *full-information* posterior mean of :math:`\boldsymbol{\theta}` as we increase :math:`L`. The paper also shows how to train a second neural net to learn the full-information posterior variance.

.. figure:: diagram_net.png
   :width: 80%
   :align: left
   :target: ../_static/diagram_net-view.html

|

Applications
------------

We provide Matlab code for two examples: 

* **A mixed logit model**. Because likelihood in mixed logit is relatively easy to simulate, full-info NNE shows no computational or accuracy advantages here, but the setting serves as a good example to show how full-info NNE works in practice.

* **A search model with unobserved consumer heterogeneity**. This examples shows the computational and accuracy advantages of full-info NNE.

You can find the code at this `TBA <https://example.com/>`_, and code documentation at the :ref:`mixed logit model <fnne_mixed_logit>` page and the :ref:`search model <fnne_search>` page.

|

.. toctree::
   :hidden:

   Overview <self>
   Mixed Logit Model <fnne_mixed_logit>
   Search Model <fnne_search>
