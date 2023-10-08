:html_theme.sidebar_secondary.remove:

:parenttoc: True

.. _code:

Code
============

|

You can find the Matlab files |code_link| (still in beta version).

.. |code_link| raw:: html

   <a href="https://github.com/pnnehome/code_matlab" target="_blank">here</a>

Below are documentation for each file. We suggest you first read the example in :ref:`home`, which is an easier way to get an idea of how the code works.

|

Current settings on the range of data accepted by pNNE:

\*Current settings are relatively restrictive as pNNE is still in beta stage. We can change these settings in the next version of pNNE if there are requests to do so. Please contact us if you want to see a setting broadened.

.. role:: brown-text

- :brown-text:`Sample size`: n ≥ 1000 number of consumers (or search sessions).
- :brown-text:`Number of options` per consumer: 15 ≤ J ≤ 35.
- Number of :brown-text:`product attributes`: 2 ≤ p ≤ 8. Number of consumer attributes:  0 ≤ c ≤ 5. Number of advertising attributes:  0 ≤ a ≤ 2. 
- :brown-text:`Buy rate` (fraction of consumers who didn't buy outside good) is between 0.5% and 50%.
- :brown-text:`Search rate` (fraction of consumers who went beyond the free search) is between 1% and 70%.
- :brown-text:`Average number of searches` per consumer is between 1 and 4.

|

``trained_nne.mat``

This file stores the trained neural net as well as some settings.

- ``nne``: a structure that stores the trained neural net, as well as some settings used in training, such as the largest number of product attributes allowed, the smallest sample size allowed, the range of search model parameters considered, etc.

|

``nne_estimate.m`` 

The main function to apply pNNE to your data.

.. code-block:: console

    result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se=false, checks=true)


Inputs:

- ``nne``: as described before, from trained_nne.mat.
- ``Y``: nJ by 2 matrix.  Y(J*(i-1)+j, 1) indicates whether consumer i searched option j. Y(J*(i-1)+j, 2) indicates whether consumer i bought option j.
- ``Xp``: nJ by p matrix. Xp(J*(i-1)+j, :) stores the product attributes of option j for consumer i.
- ``Xa``: nJ by a matrix. Xa(J*(i-1)+j, :) stores the marketing attributes of option j for consumer i.
- ``Xc``: n by c matrix. Xc(i, :) stores the consumer attributes of consumer i.
- ``consumer_idx``: nJ by 1 vector.  consumer_idx(J*(i-1)+1:J*i) = i.
- ``se=false``: optional input. Set to true to calculate bootstrap standard errors.
- ``checks=true``: optional input. Set to false to omit all sanity checks.

Outputs:

- ``result``: a table listing parameters and their estimates. If se=false, it has an additional column of standard errors.

|

``moments.m`` 

This function is used by nne_estimate.m. The function computes summary moments from data. These summary moments 
are then fed into the neural net as inputs.

|

``regress_coef.m`` 

This function is used by moments.m. The function does ridge linear, logit, and multinomial logit regressions. 
These regressions are applied to the data to produce some of the summary moments.

|

``mseLayer.m`` 

This function is a custom neural net layer used by our trained neural net. It is basically the same as 
Matlab's built-in regressionLayer.m, except that it has some minor customization to better suit pNNE training.

|

``search_stat.m`` 

This function calculates some summary statistics of the data. It is used by nne_estimate.m to perform 
a part of the data saneity checks.

|

``curve.mat`` 

This file stores the relation describing how log search cost affects reservation utility.

- curve: a table that stores a grid of log search cost and corresponding reservation utility.

|

``search_model.m`` 

The function that codes the search model. This function is used in training but not in estimation. 
We provide it for reference purpose. It is also useful if you want to conduct Monte Carlo experiments.

.. code-block:: console

    Y = search_model(par, curve, Xp, Xa, Xc, consumer_idx)

Inputs:

- ``par``: vector of parameter value. First a+1 entries are α. Next c+1 entries are η. Last p entries are β.
- ``curve``: as described before, from curve.mat.
- ``Xp``, Xa, Xc, and consumer_idx: as described before.

Outputs

- ``Y``: as described above. Simulated search and purchase choices.

