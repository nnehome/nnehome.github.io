:parenttoc: True

.. _nne_search:

Search model
==================

|

Below is the documentation for the Matlab code in "NNE_search" folder at this `GitHub repository <https://github.com/nnehome/nne-matlab-code>`_. The code uses NNE to estimate a consumer search model. You are welcome to modify the code to estimate your own structural model.


Workflow
----------

The following commands run a Monte Carlo experiment that estimates the search model from a simulated dataset.

.. code-block:: console

    >> monte_carlo_data		% simulate a dataset and save it in data.mat
    >> nne_gen			% generate the training examples for NNE and save them in nne_training.mat
    >> nne_train		% train a neural net and then apply it on data.mat


Description of each file
--------------------------

``model_seq_search.m``
""""""""""""""""""""""""""

This function codes a sequential search model.

.. code-block:: console

    [yd, yt, order] = model_seq_search(pos, z, consumer_id, theta, curve)

* Inputs:

  * ``pos``: product ranking positions (which affects search costs)
  * ``z``: other product attributes (e.g., review rating, price)
  * ``consumer_id``: indices of consumers
  * ``theta``: search model parameter vector
  * ``curve``: lookup table between reservation utility and search cost, available from ``curve_seq_search.csv``
 
* Outputs:

  * ``yd``: dummies indicating searches
  * ``yt``: dummies indicating purchases
  * ``order``: search order


``moments.m``
""""""""""""""""""""""""""

This function summarizes data into a set of moments.

.. code-block:: console

    output = moments(pos, z, consumer_id, yd, yt)
    
* Inputs: as described above for ``model_seq_search.m``.

* Output: a vector collecting the moments.


``normalRegressionLayer.m``
""""""""""""""""""""""""""""

This file codes the cross-entropy loss. This custom loss function is needed if we want NNE to output variance estimates in addition to point estimates.


``monte_carlo_data.m``
""""""""""""""""""""""""""

This script creates a dataset for Monte Carlo experiment. It uses ``model_seq_search.m`` to simulate the dataset under a "true" search model parameter vector, and then saves the dataset to ``data.mat``.


``nne_gen.m``
""""""""""""""""""""""""""

This script generates the training and validation examples.

* It loads the product attributes (``z`` and ``pos``)  in ``data.mat``.
* It uses ``model_seq_search.m`` to simulate the consumer choices in each training or validation example.
* It uses ``moments.m`` to summarize data in each training or validation example.
* The training and validation examples are saved to ``nne_training.mat``.


``nne_train.m``
""""""""""""""""""""""""""

This script trains a shallow neural net.

* It loads the training and validation examples from ``nne_training.mat`` (saved by ``nne_gen.m``).
* It uses ``normalRegressionLayer.m`` for the cross-entropy loss.
* Validation loss is reported. We can use this loss to choose neural net hyperparameters (e.g., the number of hidden nodes).
* It applies the trained neural net to ``data.mat``.

|

