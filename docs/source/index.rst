:html_theme.sidebar_secondary.remove:

.. _family-home:

Welcome to NNE
==============

This website provides the code and some documentation for the neural net estimator (NNE), a general
approach that uses neural nets to estimate structural econometric models.

Please select the type of NNE below.

.. raw:: html

   <div class="method-cards">

     <div class="method-card accent-nne">
       <h3>Original NNE (limited-information)</h3>
       <p>Based on &ldquo;Estimating Parameters of Structural Models Using Neural Nets,&rdquo;
          <a class="inline-link" href="https://pubsonline.informs.org/doi/10.1287/mksc.2022.0360"
             target="_blank" rel="noopener">Wei and Jiang (2025), Marketing Science</a>, 44(1).
          This NNE uses researcher-specified moments as input to the neural net. It is most useful
          when the researcher has clear intuition about what data moments identify the structural
          model.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="nne/index.html" aria-label="Open Original NNE"></a>
     </div>

     <div class="method-card accent-pnne">
       <h3>Pre-trained NNE</h3>
       <p>Based on &ldquo;Pre-Training Estimators for Structural Models: Application to Consumer
          Search,&rdquo; <a class="inline-link" href="https://arxiv.org/abs/2505.00526"
          target="_blank" rel="noopener">Wei and Jiang (2025)</a>. This NNE pretrains a neural net
          for a given structural model, so researchers can use it to estimate the structural model
          right away &mdash; as easy as running a reduced-form regression.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="pnne/index.html" aria-label="Open Pre-trained NNE"></a>
     </div>

     <div class="method-card accent-fnne">
       <h3>Full-information NNE</h3>
       <p>Based on &ldquo;Estimating and Assessing Identification of Structural Models via Deep
          Learning,&rdquo; <a class="inline-link"
          href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6774178"
          target="_blank" rel="noopener">Wei and Jiang (2026)</a>. This NNE uses the whole dataset
          (instead of researcher-specified moments) as input. It can automatically exploit variation
          in data and thus is useful for not only estimating but also assessing the identification of
          a structural model.</p>
       <span class="method-open">Open &rarr;</span>
       <a class="card-stretch" href="fnne/index.html" aria-label="Open Full-information NNE"></a>
     </div>

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
