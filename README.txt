Neural Net Estimators (NNE) — Documentation Website
===================================================

About
-----
This is the documentation site for the neural net estimator (NNE) family of
estimators for structural econometric models. It collects three related
methods, each as its own section reachable from the navigation bar:

  - Original NNE        — the core, moment-based estimator
  - Pre-trained NNE     — a ready-to-use estimator for consumer search
  - Full-information NNE — uses the whole dataset and assesses identification

The site is built with Sphinx and the pydata-sphinx-theme, and is deployed to
https://nnehome.github.io via GitHub Actions on every push to the main branch.


Where the content lives
------------------------
All source files are under docs/source/ :

  index.rst              The landing page (method cards + comparison table).
  nne/                   Original NNE: index.rst (overview) + code pages.
  pnne/                  Pre-trained NNE: index.rst (overview) + code/data/contact.
  fnne/                  Full-information NNE: index.rst (overview) + code pages.
  _static/custom.css     Site styling (layout, navbar dropdowns, tables).
  _static/*.svg          Figures (e.g. the architecture diagram).
  _templates/            Navbar dropdowns and previous/next/Home footer.
  conf.py                Sphinx configuration (theme, extensions, options).

Pages are written in reStructuredText (.rst). To edit text, change the
relevant .rst file; to change look-and-feel, edit _static/custom.css.


Run / build locally
--------------------
First time only — create a virtual environment and install dependencies:

  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Build the site:

  cd docs
  make html

Then open docs/build/html/index.html in a browser.

Note: when you change ONLY CSS or templates (not a .rst file), Sphinx may skip
copying static files. Force a clean rebuild in that case:

  rm -rf build && make html
