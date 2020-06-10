import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


project = "kprototypes"
# copyright = "2020, Johan Berdat"
author = "Johan Berdat"

release = "0.1.2"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_show_copyright = False
