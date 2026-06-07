:parenttoc: True

.. _pnne_code:

Code
====

|

You can find the Matlab (R2024b) files in this
`GitHub directory <https://github.com/pnnehome/code_matlab>`__ (we plan to provide Python/R code in
the near future). Explanation for each file is provided below. However, we suggest first reading the
example on the :ref:`home <pnne>` page, which is an easier way to get an overall idea of how the code
works.

|

Range of data accepted
-----------------------

.. raw:: html

   <p>Below list the data dimensions currently accepted by the main file <code class="docutils literal notranslate"><span class="pre">nne_estimate.m</span></code>. (<span class="note-text">Note: current settings may seem restrictive, but we can adjust them in the future if needed &mdash; feel free to contact us.</span>)</p>

* **Sample size:** :math:`n` ≥ 1000 consumers (or search sessions).

* **Number of options**: 15 ≤ :math:`J` ≤ 35.

* Number of **product attributes** is ≥ 2 and ≤ 8.

* Number of **consumer attributes** is ≤ 5.

* Number of **advertising attributes** is ≤ 2.

In addition, the NNE is pretrained on data with following characteristics. If your data's
characteristics fall outside these ranges, the pretrained NNE may not work as intended.

* **Buy rate** (fraction of consumers who bought inside good) is 0.5% ~ 70%.

* **Search rate** (fraction of consumers who went beyond free search) is 1% ~ 80%.

* **Average number of searches** per consumer is 1 ~ 6.

The attributes, ``Xp``, ``Xa``, ``Xc``, need not be standardized. However, we recommend de-meaning
them which makes the intercepts (:math:`\alpha_0, \eta_0`) easier to interpret. We also recommend the
usual practices to treat outliers (e.g., winsorizing, transformation) before using the pretrained NNE.

|

Description of files
--------------------

``trained_nne.mat``
~~~~~~~~~~~~~~~~~~~~

This file stores the trained neural net as well as some pre-defined settings.

* ``nne``: a structure that stores the trained neural net, as well as some settings used in training, such as the range of data dimensions and the prior of the search model parameter.

``nne_estimate.m``
~~~~~~~~~~~~~~~~~~

This is the main function that applies the pretrained NNE to your data.

.. code-block:: console

   result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = false, checks = true)

Inputs:

* ``nne``: already described above, from ``trained_nne.mat``.

* ``Y``: :math:`nJ` by 2 matrix. The :math:`((i-1)J+j)`-th row corresponds to product :math:`j` for consumer :math:`i`. The first value indicates if :math:`j` is searched, and the second value indicates if :math:`j` is bought.

* ``Xp``: a matrix with :math:`nJ` rows. The :math:`((i-1)J+j)`-th row stores the product attributes of product :math:`j` for consumer :math:`i`.

* ``Xa``: a matrix with :math:`nJ` rows. The :math:`((i-1)J+j)`-th row stores the advertising attributes of product :math:`j` for consumer :math:`i`.

* ``Xc``: a matrix with :math:`n` rows. The :math:`i`-th row stores the consumer attributes of consumer :math:`i`.

* ``consumer_idx``: a column vector with :math:`nJ` values. The :math:`((i-1)J+j)`-th value equals :math:`i`.

* ``se = false``: optional input. Set it to "true" to calculate bootstrap standard errors.

* ``checks = true``: optional input. Set it to "false" to omit all sanity checks.

Output:

* ``result``: a table showing parameter estimate and bootstrap standard errors (if selected).

``moments.m``
~~~~~~~~~~~~~

This function is used by ``nne_estimate.m``. It computes data moments. The moments include summary
statistics as well as coefficients of reduced-form regressions. These data moments are fed into the
neural net as inputs.

``reg_logit.m``
~~~~~~~~~~~~~~~

This function is used by ``moments.m``. It performs ridge logit or multinomial-logit regressions. It
runs faster than Matlab built-in regressions. The speedup is helpful for pretraining but less
important when applying the pretrained NNE. Nevertheless, we make use of it in both pretraining and
applications.

``reg_linear.m``
~~~~~~~~~~~~~~~~

This function is used by ``moments.m``. It performs ridge linear regression.

``data_checks.m``
~~~~~~~~~~~~~~~~~

This function is used by ``nne_estimate.m`` to run some basic sanity checks on data. For example, did
every consumer buy at most one option; did every consumer conduct the free search?

.. raw:: html

   <p>(<span class="note-text">Note: The following files are not required to apply the pretrained NNE. They are provided for reference purposes.</span>)</p>

``curve.mat``
~~~~~~~~~~~~~

This file stores a lookup table for the relation between search cost and reservation utility. This
relation is useful for computing the optimal consumer choices in the sequential search model.

``search_model.m``
~~~~~~~~~~~~~~~~~~

This function codes the search model. The function is used in pretraining but not in application. We
provide it for reference purpose. It is also useful if you want to conduct Monte Carlo experiments.

.. code-block:: console

   Y = search_model(par, curve, Xp, Xa, Xc, consumer_idx)

Inputs:

* ``par``: vector of the parameter value for the search model.

* ``curve``: a lookup table as described above, from ``curve.mat``.

* ``Xp``, ``Xa``, ``Xc``, and ``consumer_idx``: data as described before.

Outputs:

* ``Y``: a matrix with the simulated search and purchase choices, formatted as described before.

``winsorize.m``
~~~~~~~~~~~~~~~

This function winsorizes data at 0.5 and 99.5 percentiles. We suggest applying it to your data before
applying the pretrained NNE. For example, ``Xp = winsorize(Xp)``.
