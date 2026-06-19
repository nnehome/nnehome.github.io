# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NNE'
copyright = "2023, Yanhao 'Max' Wei, Zhenling Jiang"
author = "Yanhao 'Max' Wei, Zhenling Jiang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_new_tab_link',
    'sphinx_copybutton',
    'sphinx_toolbox.collapse',
    'sphinx_reredirects'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = '_static/logo.png'

# No left sidebar anywhere: each section's pages are reached through the
# navbar dropdowns (see _templates/navbar-nav.html), which frees the full
# width for the main content.
html_sidebars = {
    "**": [],
}

html_theme_options = {
    "navbar_align": "content",
    "navbar_persistent": ["search-button"],
    "logo": {
        "text": "NNE",
    },
    "show_prev_next": False,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nnehome",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_show_sourcelink = False

redirects = {
    "home/home.html": "../index.html",
}

rst_prolog = """
.. role:: note-text

.. role:: raw-html(raw)
   :format: html
"""
