:parenttoc: True

.. _pnne_code:

Code
====

|

Below is documentation for the Matlab (2024b) files at this `GitHub directory <https://github.com/pnnehome/code_matlab>`__, which provide a pretrained NNE of a consumer search model. We suggest taking a look at the guide on the :ref:`home <pnne>` page before reading this documentation.

|

Description of files
--------------------

``trained_nne.mat``
~~~~~~~~~~~~~~~~~~~~

This file stores the pretrained neural net as well as some pre-defined settings.

* ``nne``: a structure that stores the trained neural net, as well as some settings used in training, such as the range of data sizes and the prior of the search model parameter.

``nne_estimate.m``
~~~~~~~~~~~~~~~~~~

This is the main function that applies the pretrained NNE to your data.

.. code-block:: console

   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = false, checks = true)

Inputs:

* ``nne``: already described above, available from ``trained_nne.mat``.

* ``Y``: :math:`nJ` by 2 matrix. The :math:`((i-1)J+j)`-th row corresponds to product :math:`j` for consumer :math:`i`. The two values in each row indicate if :math:`j` is searched and if :math:`j` is bought, respectively.

* ``Xp``: a matrix with :math:`nJ` rows. The :math:`((i-1)J+j)`-th row stores the product attributes of product :math:`j` for consumer :math:`i`.

* ``Xa``: a matrix with :math:`nJ` rows. The :math:`((i-1)J+j)`-th row stores the advertising attributes of product :math:`j` for consumer :math:`i`.

* ``Xc``: a matrix with :math:`n` rows. The :math:`i`-th row stores the consumer attributes of consumer :math:`i`.

* ``consumer_idx``: a column vector with :math:`nJ` values. The :math:`((i-1)J+j)`-th value equals :math:`i`.

* ``se = false``: optional input. Set it to "true" to calculate bootstrap standard errors.

* ``checks = true``: optional input. Set it to "false" to omit all sanity checks.

Output:

* ``result``: a table showing the parameter estimates.

``moments.m``
~~~~~~~~~~~~~

This function computes data moments and is used by ``nne_estimate.m``. The moments include summary
statistics as well as coefficients of reduced-form regressions. These moments are fed into the
neural net as input.

``reg_logit.m``
~~~~~~~~~~~~~~~

This function runs ridge logit or multinomial-logit regressions and is used by ``moments.m``. It
is faster than Matlab built-in regressions. The speedup substantially reduces the pretraining time (albeit is less
noticeable when we apply the pretrained NNE).

``reg_linear.m``
~~~~~~~~~~~~~~~~

This function runs ridge linear regression and is used by ``moments.m``.

``data_checks.m``
~~~~~~~~~~~~~~~~~

This function is used by ``nne_estimate.m`` to run some basic sanity checks on data. For example,
every consumer can buy at most one option; every consumer should conduct the free search.

|

:note-text:`Note\: The following files are used in pretraining, but not required to apply the pretrained NNE. We provide them for reference purposes.`

``curve.mat``
~~~~~~~~~~~~~

This file stores a lookup table of the relation between search cost and reservation utility. This
relation is used for computing the optimal consumer choices in the sequential search model.

``search_model.m``
~~~~~~~~~~~~~~~~~~

This function codes the search model.

.. code-block:: console

   Y = search_model(par, curve, Xp, Xa, Xc, consumer_idx)

Inputs:

* ``par``: the search model parameter vector.

* ``curve``: a lookup table described above, available from ``curve.mat``.

* ``Xp``, ``Xa``, ``Xc``, and ``consumer_idx``: data as described before.

Outputs:

* ``Y``: searches and purchases, formatted as described before.

``winsorize.m``
~~~~~~~~~~~~~~~

This function winsorizes data at 0.5 and 99.5 percentiles. We suggest using it on your data before
applying the pretrained NNE. For example, ``Xp = winsorize(Xp)``.
