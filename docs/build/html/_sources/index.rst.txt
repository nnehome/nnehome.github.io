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
    Code <code/code>
    Contact <contact/contact>

Welcome to pNNE
===============

|

This website provides a pre-trained estimator for consumer search model.

The estimator is built as pNNE (pre-trained neural net estimator). The idea is to allocate the bulk of work to pre-training. So an end user can estimate the structural econometric model as easy as a reduced-form model.

Below is a brief overview how to apply pNNE to your search data. Click "code" tab above for Matlab code files.

|

The main function to execute estimation is ``nne_estimate.m``, as shown below.

.. code-block:: console

   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

The required inputs: (i) ``nne`` stores the trained neural net and some pre-defined settings, available from the file ``trained_nne.mat``, and (ii)  ``Xp``, ``Xa``, ``Xc``, ``Y``, and ``consumer_index`` store your data.

The output: ``result`` is a table storing the parameter estimate of a standard consumer search model.

Below shows an example. The total execution time is 0.78 sec on a laptop, including overheads such as some data sanity checks.

.. code-block:: console

   >>  load('trained_nne.mat', 'nne')
   >>  tic 
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

The rest of this page gives more details on the search model, how to format data (``Xp``, ``Xa``, ``Xc``, ``Y``, and ``consumer_index``), and how to get standard errors.

|

1) The search model
-------------------
The econometric model to be estimated is a standard sequential search model. Each consumer faces J products (or options). She decides which options to search and which option to buy. Searching a product incurs a search cost but also reveals some utility for the product.

A typical setting for the model is consumer search online, where a "consumer" is a "search session." A consumer is shown a list of products. Clicking a product for more information is a search. After searching, the consumer may buy one product or an outside good. The list page usually shows some product attributes (e.g., review rating) that affect consumer utility for products. In addition, there may be some consumer attributes (e.g., past spending, visit time) that affect consumer outside utility. There may also be advertising attributes (e.g., ranking in the list) that affect search costs. 

There are three sets of parameters: :math:`\beta` describes how product attributes affect utility,  :math:`\eta` describes how consumer attributes affect outside utility, and :math:`\alpha` describes how advertising attributes affect search costs.

A free search is endowed, to accommodate most data where consumer searches at least once.

|

2) Data format
-----------------

Oraganize data into five arrays: ``Xp``, ``Xa``, ``Xc``, ``Y``, and ``consumer_index``. Respectively: ``Xp`` stores product attributes; ``Xa`` stores advertising attributes; and ``Xc`` stores consumer attributes; ``Y`` stores search and purchase outcomes; ``consumer_index`` identifies consumers.

In the example below, there are n=10,000 consumers (or search sessions) and each consumer has J=20 options for search. There are 2 product attributes, 1 advertising attribute, and 3 consumer attributes.

.. code-block:: console

   >>  whos Xp Xa Xc Y consumer_idx

       Name               Size             Bytes         Class      Attributes

       Xa                200000x1         1600000        double 
       Xc                 10000x3          240000        double 
       Xp                200000x2         3200000        double              
       consumer_idx      200000x1         1600000        double              
       Y                 200000x2         3200000        double 

Below previews the first rows of  ``Xp``, ``Xa``, ``Y``, and ``consumer_index``. Variable ``Y`` has two columns; the 1st column indicates search and 2nd column indicates purchase. We see that the first consumer searched option 1, 3, 4, and 8. She bought option 3.

``Xc``, not displayed here, has only 10,000 rows, same as the number of consumers.

If data did not feature advertising attributes, we'd let ``Xa`` be 200000 by 0 (i.e., empty). If no consumer attributes, we'd let ``Xc`` be 10000 by 0.

.. code-block:: console

   >>  table(consumer_idx, Y, Xp, Xa)
   ans = 200000×4 table
    consumer_idx      Y             Xp          Xa
    ____________    ______    ______________    __
           1        1    0    4      0.67743    0 
           1        0    0    5       1.1052    0 
           1        1    1    1     -0.24542    0 
           1        1    0    5      0.78452    0 
           1        0    0    4      0.10519    0 
           1        0    0    5     0.068463    0 
           1        0    0    4      0.35691    1 
           1        1    0    5      0.61307    0 
           1        0    0    3       1.1809    1 
           1        0    0    5      0.91391    0 
           1        0    0    5     0.054537    0 
           1        0    0    3       1.0015    0 
           1        0    0    5      0.73938    0 
           1        0    0    3    -0.020808    0 
           1        0    0    5      0.23587    0 
           1        0    0    5      0.43236    1 
           1        0    0    5    -0.040968    0 
           1        0    0    5       0.4916    0 
           1        0    0    2      0.96498    0 
           1        0    0    2      0.46767    0 
           2        1    0    3      0.72159    1 
           2        0    0    5     -0.39847    0 
           2        0    0    5      0.73669    1 
         :            :             :           : 

|

3) Bootstrap standard errors
----------------------------
The nne_estimate function has standard error calculation built in. Simply add ``"se = true"`` option as shown below. The output will include an additional column of standard errors. The calculation bootstraps 50 samples so execution time will be longer, but it can take advantage of parallel computing toolbox if installed.

.. code-block:: console

    >> result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = true);
    >> result
    result = 8×3 table

       name          val          se   
    __________    _________    ________
    "\alpha_0"      -6.4546    0.071848
    "\alpha_1"     -0.11333    0.044997
    "\eta_0"          5.929    0.061318
    "\eta_1"      -0.048687    0.011513
    "\eta_2"        0.22004    0.029282
    "\eta_3"       0.052894    0.023437
    "\beta_1"       0.25836    0.004951
    "\beta_2"      -0.53811    0.011032

