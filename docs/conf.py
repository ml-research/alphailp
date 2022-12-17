# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# sys.path.insert(0, '../src')

sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../src'))

project = 'alphailp'
copyright = '2022, HikaruShindo'
author = 'HikaruShindo'

version = 'v1.0'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True


html_static_path = ['_static']
html_logo = "_static/aILP_logo_pink.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
html_favicon = "_static/alphailp_favicon.png"


html_sidebars = {'**': ['globaltoc.html',
                        'relations.html', 'sourcelink.html', 'searchbox.html']}
