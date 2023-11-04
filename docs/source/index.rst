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

Welcome to NNE
===============

|

This website describes the neural net estimator (NNE) to estimate structural models, as proposed 
in Wei and Jiang (2023). It provides a computationally-light alternative to simulated maximum likelihood 
or simulated method of moments. NNE is especially suitable for cases where many simulations are needed to 
evaluate likelihood/moment functions.

|

Overview of NNE
---------------
We write down a structural model: :math:`{y = g(x, ϵ; θ)}`. The goal of estimation is to obtain the parameter :math:`{θ}` 
after observing the covariates :math:`{x}` and outcome :math:`{y: {y,x} → θ}`.

The key idea of NNE is to use neural nets to directly learn the mapping from data to parameters. 
The graph below provides an overview of NNE.

.. math::
   :label: neural-net-training

   \begin{align*}
   \text{train a } \text{neural net } f(\cdot)
   \begin{cases}
   \boldsymbol{\theta}^{(1)} \xrightarrow{\boldsymbol{g}(\boldsymbol{x}_{i},\boldsymbol{\varepsilon}_{i}^{(1)};\boldsymbol{\theta}^{(1)})} & \{\boldsymbol{y}_{i}^{(1)},\boldsymbol{x}_{i}\}_{i=1}^{n} \xrightarrow{\text{moments}} \boldsymbol{m}^{(1)} \xrightarrow{\text{neural net}} \widehat{\boldsymbol{\theta}}^{(1)} \\
   \boldsymbol{\theta}^{(2)} & \{\boldsymbol{y}_{i}^{(2)},\boldsymbol{x}_{i}\}_{i=1}^{n} \xrightarrow{\text{moments}} \boldsymbol{m}^{(2)} \xrightarrow{\text{neural net}} \widehat{\boldsymbol{\theta}}^{(2)} \\
   \vdots & \vdots \\
   \boldsymbol{\theta}^{(L)} & \{\boldsymbol{y}_{i}^{(L)},\boldsymbol{x}_{i}\}_{i=1}^{n} \xrightarrow{\text{moments}} \boldsymbol{m}^{(L)} \xrightarrow{\text{neural net}} \widehat{\boldsymbol{\theta}}^{(L)}
   \end{cases}
   \end{align*}

.. math::
   :label: neural-net-application

   \begin{align*}
   \text{apply } f(\cdot) \text{on real data}
   \begin{cases}
   \{\underbrace{\boldsymbol{y}_{i},\boldsymbol{x}_{i}}_{\text{real data}}\}_{i=1}^{n} \xrightarrow{\text{moments}} \boldsymbol{m} \xrightarrow{\text{neural net}} \underbrace{\widehat{\boldsymbol{\theta}}}_{\text{estimate}}
   \end{cases}
   \end{align*}

Notation:
 :math:`\boldsymbol{\theta}^{(\ell)}` drawn from a space :math:`\Theta`;
 :math:`\boldsymbol{y}_{i}=\boldsymbol{g}(\boldsymbol{x}_{i},\boldsymbol{\varepsilon}_{i};\boldsymbol{\theta})` a structural model;
 :math:`\boldsymbol{y}^{(\ell)}` simulated outcome;
 :math:`\boldsymbol{m}^{(\ell)}` simulated moments;
 :math:`\widehat{\boldsymbol{\theta}}` neural net prediction




1. We draw parameter values :math:`\theta^{(l)}` uniformly from a parameter space :math:`\Theta`. Using the structural model, we can generate the outcome :math:`y^{(l)}` under :math:`\theta^{(l)}`. After repeating this procedure a number of times, we get the corresponding datasets that are generated under a range of parameter values. These datasets form the basis of the training examples where we can use to learn the mapping from data to the "correct" parameter values.

2. To make training easier, we can summarize the data :math:`\{y^{(l)}, x\}`  into data moments :math:`m^{(l)}`.

3. The neural net takes the data moments as input and predicts the parameter value underlying that dataset.

4. Once the neural net is trained, we plug in the real data moments to obtain NNE estimates :math:`\hat{\theta}`.

The neural net can output "standard errors" in addition to point estimates. We establish that this neural net estimator (NNE)
converges to limited-information Bayesian posterior when the number of training datasets L is sufficiently large. 
Besides the benefit of light computational cost, NNE is also robust to redundant moments, which is beneficial for cases where 
there lacks clear guidance on moment choices from a theoretical perspective. 

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

