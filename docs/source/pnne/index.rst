.. _pnne:

Pre-trained NNE
===============

|

This page describes the **pre-trained neural net estimator (pre-trained NNE)** for the consumer
sequential search model, based on Wei and Jiang (2025).

Unlike :ref:`NNE <nne>` and :ref:`full-information NNE <fnne>`, which you train yourself, the
pre-trained NNE is **ready to use**. We have already trained the neural net, so you only need to plug
in your data. The core function ``nne_estimate.m`` takes the trained net plus your data and returns
parameter estimates in **under a second**.

.. note::

   **Migrated section.** This content was previously hosted at ``pnnehome.github.io``. The code and
   data pages below summarize that documentation; copy any exact code listings from the
   `pre-trained NNE GitHub repository <https://github.com/nnehome>`_ as needed.

|

The search model
-----------------

The estimator targets a **sequential search model**. The parameters fall into three groups:

* :math:`\boldsymbol{\alpha}` (e.g., :math:`\alpha_0, \alpha_1`): effects of **advertising** attributes on search costs.
* :math:`\boldsymbol{\beta}` (e.g., :math:`\beta_1, \beta_2`): effects of **product** attributes on consumer utility.
* :math:`\boldsymbol{\eta}` (e.g., :math:`\eta_0, \dots, \eta_3`): effects of **consumer** attributes on outside utility.

|

How to use it
-------------

Organize your data into the arrays described on the :ref:`Data <pnne_data>` page, then call the
estimator:

.. code-block:: console

   >> result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

This returns a table of parameter estimates. To also obtain **bootstrap standard errors**, set
``se = true`` (50 bootstrap resamples; supports parallel computing):

.. code-block:: console

   >> result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = true)

See the :ref:`Code <pnne_code>` page for the full list of arguments and supporting files.

|

|

Papers
------

* Wei and Jiang (2025) — pre-trained NNE for consumer search.
* Wei and Jiang (2024) "Estimating Parameters of Structural Models with Neural Networks," *Marketing Science*. `SSRN link <https://ssrn.com/abstract=3496098>`_
* Ursu, Seiler, and Honka (2025) — survey on sequential search applications.

|

.. toctree::
   :hidden:

   Overview <self>
   Code <code>
   Data <data>
   Contact <contact>
