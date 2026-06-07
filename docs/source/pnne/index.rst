.. _pnne:

Welcome to Pre-trained NNE
==========================

.. raw:: html

   <p>(<span class="note-text">Note: This website is being updated over time. Last update on June 5, 2025.</span>)</p>

This is the companion website for `Wei and Jiang (2025) <https://arxiv.org/abs/2505.00526>`__. It provides a pretrained estimator for a
consumer search model used in economics & marketing. The estimator is based on a neural net that
recognizes search model parameters from data patterns. The neural net is pretrained so the estimation
cost for users is minimal. We call this estimator a **pretrained neural net estimator (NNE)**.
Pretraining NNE is generally applicable to structural models, though here we focus on the search model.

Below is a brief overview of how to apply the pretrained NNE to your search data. For Matlab code and
detailed documentation, see the :ref:`code <pnne_code>` page. For examples to try out, see the
:ref:`data <pnne_data>` page.

|

Overview
--------

The main function to execute is ``nne_estimate.m``, as shown below.

.. code-block:: console

   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

Explanation for inputs and output:

* Input ``nne`` stores the trained neural net and some pre-defined settings, available from the file ``trained_nne.mat``.

* Inputs ``Xp``, ``Xa``, ``Xc``, ``Y``, and ``consumer_idx`` are your data (more on them below).

* Output ``result`` is a table with the parameter estimate for the search model.

Below shows an example. The total execution time is 0.78 sec on a laptop. The time includes overheads
such as some data sanity checks.

.. code-block:: console

   >> load('trained_nne.mat', 'nne')
   >> tic
   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx);
   toc
   Elapsed time is 0.784314 seconds.
   >> result
   result = 8×2 table
       name           val
    ----------      -------
    "\alpha_0"      -6.4546
    "\alpha_1"     -0.11333
    "\eta_0"          5.929
    "\eta_1"      -0.048687
    "\eta_2"        0.22004
    "\eta_3"       0.052894
    "\beta_1"       0.25836
    "\beta_2"      -0.53811

The rest of this page gives details on: (i) the search model, (ii) how to format data, and (iii) how
to get standard errors.

|

1) The search model
~~~~~~~~~~~~~~~~~~~~~

Sequential search models have been commonly applied in economics & marketing. A consumer faces
:math:`J` products (or options). She decides which options to search and which option to buy.
Searching an option incurs a search cost but also reveals some utility for that option. The search
model estimated by our pretrained NNE here is a standard one. The exact specification of it is given
in Wei and Jiang (2025). Here we give a high-level description:

* A consumer faces a list of :math:`J` products plus an outside good. A free search is endowed (because in most data each consumer searches at least once).

* The list shows some **product attributes** that affect consumer's utility for products. Their effects are described by :math:`\beta`.

* There may be some **consumer attributes** that affect consumer's outside utility. Their effects are described by :math:`\eta`.

* There may be **advertising attributes** that affect search costs (but not utility). Their effects are described by :math:`\alpha`.

For a survey on empirical applications of search models, see `Ursu, Seiler, & Honka (2025) <https://link.springer.com/article/10.1007/s11129-024-09291-2>`__.

|

2) Data format
~~~~~~~~~~~~~~~

Our code assumes that the data are represented in five arrays: ``Y``, ``Xp``, ``Xa``, ``Xc``, and
``consumer_idx``. The easiest way to explain them is through an example of shopping at an online
retailer, which we illustrate below. There are :math:`n = 10{,}000` consumers and :math:`J = 15`
options per consumer.

* ``consumer_idx`` is a column vector with :math:`nJ` rows. Each value indicates the consumer.

* ``Y`` has two columns with :math:`nJ` rows; the 1st column indicates searches and 2nd column indicates purchases. In this example, we see the first consumer searched the 1st, 3rd, and 8th options. She bought the 3rd option.

* ``Xp`` stores product attributes. In this example, there are two product attributes: a 1-5 product rating and log price.

* ``Xa`` stores advertising attributes. In this example, the single advertising attribute indicates whether the product is highlighted on the retailer website.

* ``Xc`` stores consumer attributes. It has only :math:`n` rows, one for each consumer. It is not shown below.

.. code-block:: console

   >> table(consumer_idx, Y, Xp, Xa)
   ans = 150000×4 table
    consumer_idx      Y             Xp          Xa
    ______________    ______    ______________    __
           1        1    0    4      0.67743    1
           1        0    0    5       1.1052    1
           1        1    1    1     -0.24542    1
           1        0    0    5      0.78452    0
           1        0    0    4      0.10519    0
           1        0    0    5     0.068463    0
           1        0    0    4      0.35691    0
           1        1    0    5      0.61307    0
           1        0    0    3       1.1809    0
           1        0    0    5      0.91391    0
           1        0    0    5     0.054537    0
           1        0    0    3       1.0015    0
           1        0    0    5      0.73938    0
           1        0    0    3    -0.020808    0
           1        0    0    5      0.23587    0
           2        0    0    3      0.72159    1
           2        1    0    5     -0.39847    1
           2        0    0    5      0.73669    1
         :            :             :           :

More generally, if your data do not feature advertising or consumer attributes, you can simply let
``Xa`` or ``Xc`` be an empty matrix.

|

3) Bootstrap standard errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nne_estimate.m`` function has standard error calculation built in. Simply add ``se = true``
option as shown below. The output will include an additional column of standard errors. The
calculation bootstraps 50 samples so execution time will be somewhat longer, but it can take advantage
of parallel computing toolbox if installed.

In this example, we see :math:`\alpha_1` is negative, indicating that the highlighted products enjoy a
lower search cost. We see :math:`\beta_1` is positive, indicating that consumer utility increases with
product rating.

.. code-block:: console

   >> result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = true)
   result = 8×3 table

      name          val          se
   ______________    _______    _______
   "\alpha_0"      -6.4546    0.071848
   "\alpha_1"     -0.11333    0.044997
   "\eta_0"          5.929    0.061318
   "\eta_1"      -0.048687    0.011513
   "\eta_2"        0.22004    0.029282
   "\eta_3"       0.052894    0.023437
   "\beta_1"       0.25836    0.004951
   "\beta_2"      -0.53811    0.011032

|

Papers
------

Wei, Yanhao 'Max' and Zhenling Jiang (2025), "Pretraining Estimators for Structural Models:
Application to Consumer Search." `arXiv <https://arxiv.org/abs/2505.00526>`__

Yanhao Wei, Zhenling Jiang (2025). "Estimating Parameters of Structural Models Using Neural Networks"
`Marketing Science <https://pubsonline.informs.org/doi/10.1287/mksc.2022.0360>`__, 44 (1), 102-128.

Ursu, Raluca, Stephan Seiler, and Elisabeth Honka (2025), "The Sequential Search Model: A Framework
for Empirical Research." `QME <https://link.springer.com/article/10.1007/s11129-024-09291-2>`__, 23(1): 165-213.

|

.. toctree::
   :hidden:

   Overview <self>
   Code <code>
   Data <data>
   Contact <contact>
