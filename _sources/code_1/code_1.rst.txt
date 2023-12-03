:parenttoc: True

.. _code_1:

Consumer search
============================

|

One example application of NNE is to estimate consumer search model. This `GitHub <https://github.com/nnehome/nne-matlab>`_ page provides the Matlab (2023b) code. Below we provide description of the code files. The code itself is commented as well. You can modify the code for your own structural model.

For more details of this application, see our paper referred in :ref:`home <home>` page.

|

An example to use the code:
----------------------------

Run the four scripts in order as follows. This example is a Monte Carlo experiment that uses NNE to estimate the search model parameter from a simulated dataset.

.. code-block:: console

    >> monte_carlo_data		% simulate a dataset, saved in data.mat
    >> nne_gen			% generate the training examples for NNE, saved in nne_training.mat
    >> nne_train		% train a neural net, saved in nne_trained.mat
    >> nne_use			% apply the trained neural net on the data in data.mat

Three functions are used in these scripts: ``model_seq_search.m``, ``moments.m``, and ``normalRegressionLayer.m``, which we explain below.
    
..
	The main code scripts are ``nne_gen.m``, ``nne_train.m``, and ``nne_use.m``. These three scripts correspond to steps 1 & 2, step 3, and step 4, respectively, in the step-by-step procedure of NNE listed on :ref:`home<home>` page. The data for estimation is stored in ``data.mat``. You can use script ``monte_carlo_data.m`` to simulate data for Monte Carlo experiments. Other files are the supporting functions used by these scripts.

|

Description of each file:
--------------------------

``model_seq_search.m``
""""""""""""""""""""""""""

This function codes a sequential search model.

.. code-block:: console

    [yd, yt, order] = model_seq_search(pos, z, consumer_id, theta, curve)

* Inputs:

  * ``pos``: product ranking positions
  * ``z``: other product attributes (e.g., review rating, price)
  * ``consumer_id``: indices of consumers (or search sessions)
  * ``theta``: vector of search model parameter
  * ``curve``: relation between search cost and reservation utility, stored in ``curve_seq_search.csv``
 
* Outputs:

  * ``yd``: dummies indicating if products are searched
  * ``yt``: dummies indicating if products are bought
  * ``order``: order of search

|

``moments.m``
""""""""""""""""""""""""""

This function summarizes data into a set of moments (used in Step 2 in the procedure on :ref:`home<home>` page).

.. code-block:: console

    output = moments(pos, z, consumer_id, yd, yt)
    
* Inputs: as described above for ``model_seq_search.m``.

* Output: a vector collecting the values of the moments.

|

``normalRegressionLayer.m``
""""""""""""""""""""""""""""

This file codes the cross-entropy loss. It extends the Matlab 's built-in MSE loss. This loss function is needed if we want NNE to output estimates of statistical accuracy in addition to point estimates.

|

``monte_carlo_data.m``
""""""""""""""""""""""""""

This script generates a Monte Carlo dataset of sequential search. It uses ``model_seq_search.m`` to simulate the search and purchase choices. The data is saved in a file ``data.mat``.

|

``nne_gen.m``
""""""""""""""""""""""""""

This script generates the training and validation examples (Steps 1 & 2 in the procedure on :ref:`home<home>` page).

* It loads the product attributes (``z`` and ``pos``)  in ``data.mat``.
* It uses ``model_seq_search.m`` to simulate the consumer choices in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* Corner examples (e.g., nobody made a purchase) are dropped.
* At the end, the training and validation examples are saved in a file ``nne_training.mat``.

|

``nne_train.m``
""""""""""""""""""""""""""

This script trains a shallow neural net (Step 3 in the procedure on :ref:`home<home>` page).

* It loads the training and validation examples from ``nne_training.mat`` (created by ``nne_gen.m``).
* It uses ``normalRegressionLayer.m`` for the cross-entropy loss.
* Validation loss is reported. You can use it to choose hyperparameters, most notably the number of hidden nodes.
* At the end, the trained neural net is saved in a file ``nne_trained.mat``.

|

``nne_use.m``
""""""""""""""""""""""""""

This script applies the trained neural net to obtain parameter estimate (Step 4 in the procedure on :ref:`home<home>` page).

* It loads the trained neural net from ``nne_trained.mat`` (created by ``nne_train.m``).
* It applies the trained neural net to the data in ``data.mat``.

|

