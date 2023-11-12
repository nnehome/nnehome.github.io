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
    'sphinx_copybutton'
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


html_theme_options = {
    "navbar_persistent":[],
    "logo": {
        "link": "https://nnehome.github.io",
        "text": "NNE"
    },
    "show_prev_next": False
}
