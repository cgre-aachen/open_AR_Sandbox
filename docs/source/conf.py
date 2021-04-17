# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'open_AR_Sandbox'
copyright = '2021, Professor Florian Wellmann'
author = 'Daniel'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['nbsphinx',
              'sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'sphinx_markdown_tables',
              'sphinx_copybutton',
              #'sphinx_gallery.gen_gallery',
              'sphinx.ext.extlinks',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
]

github_url = 'https://github.com/cgre-aachen/open_AR_Sandbox'

html_context = {'source_url_prefix': "https://github.com/cgre-aachen/open_AR_Sandbox",
                'display_github': True, # Add 'Edit on Github' link instead of 'View page source'
                #'last_updated': True,
                #'commit': False,
                #'github_repo': "https://github.com/cgre-aachen/open_AR_Sandbox",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
