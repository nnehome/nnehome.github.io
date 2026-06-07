:parenttoc: True

.. _pnne_code:

Code
====

|

The pre-trained NNE is provided as **Matlab (R2024b)** files in the
`GitHub repository <https://github.com/nnehome>`_. Python/R implementations are planned for a future
release.

|

Data requirements
-----------------

The pre-trained net was trained for data within the following ranges. Estimation works best when your
data fall inside them.

.. list-table::
   :widths: 60 40
   :header-rows: 1
   :class: table-header-centered

   * - Dimension
     - Range
   * - Sample size :math:`n`
     - :math:`\geq 1{,}000` consumers
   * - Number of options :math:`J`
     - :math:`15 \leq J \leq 35`
   * - Product attributes :math:`K_p`
     - 2 – 8
   * - Consumer attributes :math:`K_c`
     - :math:`\leq 5`
   * - Advertising attributes :math:`K_a`
     - :math:`\leq 2`
   * - Buy rate
     - 0.5% – 70%
   * - Search rate
     - 1% – 80%
   * - Average searches per consumer
     - 1 – 6

We recommend **de-meaning** attributes and applying outlier treatment (winsorizing, transformation)
before estimation.

|

The ``nne_estimate`` function
-----------------------------

.. code-block:: console

   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = false, checks = true)

* Inputs:

  * ``nne``: trained neural network structure, loaded from ``trained_nne.mat``.
  * ``Y``: :math:`(n \times J) \times 2` matrix; row :math:`((i-1)J+j)` contains the ``[searched, bought]`` indicators for product :math:`j` of consumer :math:`i`.
  * ``Xp``: :math:`(n \times J) \times K_p` matrix of product attributes.
  * ``Xa``: :math:`(n \times J) \times K_a` matrix of advertising attributes.
  * ``Xc``: :math:`n \times K_c` matrix of consumer attributes.
  * ``consumer_idx``: column vector mapping each row to consumer :math:`i`.
  * ``se``: boolean; set ``true`` for bootstrap standard errors (50 resamples, parallel-computing capable).
  * ``checks``: boolean; set ``false`` to skip the data sanity checks.

* Output:

  * ``result``: a table of parameter estimates (and bootstrap standard errors, if ``se = true``).

See the :ref:`Data <pnne_data>` page for how to construct these arrays.

|

Supporting files
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :class: table-header-centered

   * - File
     - Purpose
   * - ``trained_nne.mat``
     - The pre-trained neural network and its settings.
   * - ``moments.m``
     - Computes summary statistics and regression coefficients.
   * - ``reg_logit.m``
     - Ridge logit / multinomial-logit regression.
   * - ``reg_linear.m``
     - Ridge linear regression.
   * - ``data_checks.m``
     - Validates data integrity.
   * - ``curve.mat``
     - Lookup table for the search-cost / reservation-utility relationship.
   * - ``search_model.m``
     - Reference implementation of the sequential search model.
   * - ``winsorize.m``
     - Applies 0.5–99.5 percentile winsorization.
