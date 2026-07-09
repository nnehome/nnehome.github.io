.. _pnne:

Pre-trained NNE
===============

|

This is the companion website for the pretrained Neural Net Estimator (pretrained NNE), based on the paper :note-text:`"Pretraining Estimators for Structural Models\: Application to Consumer Search."` 


1. Overview
--------------

We provide a pretrained estimator for a consumer search model used in economics & marketing. The estimator is based on a neural net that can recognize the search model parameter from data patterns. The neural net is pretrained so *the estimation
cost for users is negligible*. The approach to pretrain NNE is generally applicable to structural models, though here we focus on the search model.

The Matlab (2024b) files for this pretrained NNE can be found at this `GitHub directory <https://github.com/pnnehome/code_matlab>`__ (:note-text:`last update on May 6, 2025`). Below is a guide of how to apply this pretrained NNE to your search data. Further documentation is given on the :ref:`code <pnne_code>` page. Example datasets are provided on the :ref:`data <pnne_data>` page.


2. Using This Pretrained NNE
-----------------------------

(a) What is the model to be estimated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our pretrained NNE estimates a sequential search model. The model's exact specification is given
in the paper. Here we give a high-level description. A consumer faces :math:`J` products plus an outside good, and decides which products to search and which product to buy. The first search is free (so the consumer searches at least once).
There are *product attributes* that affect the consumer's utility for each product, and these effects are captured by :math:`\boldsymbol{\beta}`. There may be *consumer attributes* that affect the consumer's outside utility, and these effects are captured by :math:`\boldsymbol{\eta}`. There may also be *advertising attributes* that affect search costs (but not utility), and these effects are captured by :math:`\boldsymbol{\alpha}`. Parameters :math:`\eta_0` and :math:`\alpha_0` capture the baseline outside utility and search cost, respectively.


(b) How to run the estimation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our pretrained NNE is written in Matlab. The main function to execute is ``nne_estimate.m``:

.. code-block:: console

   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

* Input ``nne`` stores the trained neural net and some pre-defined settings, available from the file ``trained_nne.mat``.

* Inputs ``Xp``, ``Xa``, ``Xc``, ``Y``, and ``consumer_idx`` are your data (more explanation below).

* Output ``result`` is a table with the estimate for the search model parameter.

Below is an example. The total execution time, including overheads such as data sanity checks, is 0.78 second on a laptop.

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


(c) How to format your data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function ``nne_estimate.m`` assumes that your data are represented in a certain way. An example is given below. This example features :math:`n = 10{,}000` consumers and :math:`J = 15` products per consumer.

.. code-block:: console

   >>  table(consumer_idx, Y, Xp, Xa)
   ans = 150000×4 table
    consumer_idx      Y             Xp          Xa
    ____________    ______    ______________    __
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

* ``consumer_idx``: a column vector with :math:`nJ` rows, listing consumer indices.

* ``Y``: two binary columns indicating searches and purchases, respectively. In the example, the first consumer searched the 1st, 3rd, and 8th products, and bought the 3rd product.

* ``Xp``: product attributes. The example features two product attributes: a 1-5 review rating and log price.

* ``Xa``: advertising attributes. The example features a single advertising attribute indicating whether the product is highlighted by the seller.

* ``Xc``: consumer attributes (not shown in the example). Unlike the other data arrays, it has :math:`n` instead of :math:`nJ` rows.

More generally, you may let ``Xa = []`` or ``Xc = []`` if your data do not feature advertising or consumer attributes. ``Xp``, ``Xa``, and ``Xc`` need not be standardized. Nevertheless, we recommend mean-centering to make the interpretation of :math:`\alpha_0` and :math:`\eta_0` easier. We also recommend treating outliers (e.g., winsorizing, transformation) before using the pretrained NNE.

Below we list the data sizes currently accepted by our pretrained NNE (:note-text:`Note: these settings can be adjusted in the future if needed --- feel free to contact us`).


* :math:`n` ≥ 1000 consumers (or search sessions);

* 15 ≤ :math:`J` ≤ 35;

* Number of product attributes ≥ 2 and ≤ 8;

* Number of consumer attributes ≤ 5;

* Number of advertising attributes ≤ 2.

In addition, the NNE is pretrained on data with the following summary statistics. If your data fall outside these ranges, the pretrained NNE may not work as intended.

* Buy rate (fraction of consumers who bought inside good) is 0.5% ~ 70%.

* Search rate (fraction of consumers who went beyond free search) is 1% ~ 80%.

* Average number of searches per consumer is 1 ~ 6.


(d) How to get standard errors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nne_estimate.m`` function has standard error calculation built in. Simply add ``se = true``
option as shown below. The output will include an additional column of standard errors. The
calculation bootstraps 50 samples, so execution time will be a bit longer, but it can take advantage
of parallel computing toolbox if installed.

In this example, the estimate of :math:`\alpha_1` is negative, indicating that the highlighted products enjoy a
lower search cost. The estimate of :math:`\beta_1` is positive, indicating that utility increases with
the review rating.

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
|

.. toctree::
   :hidden:

   Overview <self>
   Code <pnne_code>
   Data <pnne_data>
