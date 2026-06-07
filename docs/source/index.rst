:html_theme.sidebar_secondary.remove:

.. _family-home:

Welcome to NNE
==============

This website provides the code and some documentation for the neural net estimator (NNE), a general
approach that uses neural nets to estimate structural econometric models.

Please select the type of NNE below.

.. raw:: html

   <div class="method-cards">

     <a class="method-card accent-nne" href="nne/index.html">
       <h3>NNE</h3>
       <p>Train a neural net on researcher-specified <em>data moments</em> to estimate your
          own structural model. The core estimator, with Matlab code for a consumer
          search model and an AR1 model.</p>
       <span class="method-paper">Wei &amp; Jiang (2024), <em>Marketing Science</em></span>
       <span class="method-open">Open &rarr;</span>
     </a>

     <a class="method-card accent-pnne" href="pnne/index.html">
       <h3>Pre-trained NNE</h3>
       <p>No training required. Plug your data into <code>nne_estimate.m</code> and get
          estimates for a sequential search model in under a second, with optional
          bootstrap standard errors.</p>
       <span class="method-paper">Wei &amp; Jiang (2025)</span>
       <span class="method-open">Open &rarr;</span>
     </a>

     <a class="method-card accent-fnne" href="fnne/index.html">
       <h3>Full-information NNE</h3>
       <p>Feed the net the <em>whole dataset</em> instead of moments. It exploits all
          variation in the data automatically &mdash; useful for estimating a model
          <em>and</em> assessing its identification.</p>
       <span class="method-paper">Wei &amp; Jiang (2025)</span>
       <span class="method-open">Open &rarr;</span>
     </a>

   </div>

|

Which one do I need?
--------------------

.. list-table::
   :widths: 28 24 24 24
   :header-rows: 1
   :stub-columns: 1
   :class: table-header-centered compare-table

   * -
     - NNE
     - Full-information NNE
     - Pre-trained NNE
   * - Input to the neural net
     - Researcher-specified moments :math:`\boldsymbol{m}`
     - The whole dataset :math:`\mathcal{D}`
     - The whole dataset :math:`\mathcal{D}`
   * - You provide
     - Your structural model + moments
     - Your structural model
     - Just your data (search model)
   * - Training needed
     - Yes (you train it)
     - Yes (you train it)
     - No &mdash; pre-trained
   * - Also gives you
     - Point estimate + accuracy
     - Identification analysis; posterior :math:`\mathrm{Var}(\boldsymbol{\theta}\mid\mathcal{D})`
     - One-call estimate + bootstrap SE
   * - Code
     - Matlab (search, AR1)
     - Matlab (mixed logit, search)
     - Matlab (``nne_estimate.m``)
   * - Paper
     - Wei & Jiang (2024)
     - Wei & Jiang (2025)
     - Wei & Jiang (2025)

|

.. toctree::
   :hidden:

   NNE <nne/index>
   Pre-trained NNE <pnne/index>
   Full-information NNE <fnne/index>
