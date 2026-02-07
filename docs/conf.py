import os

project = "trx-cpp"
author = "trx-cpp contributors"

extensions = [
    "breathe",
    "exhale",
    "sphinx.ext.autosectionlabel",
]

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

breathe_projects = {
    "trx-cpp": os.path.join(os.path.dirname(__file__), "_build", "doxygen", "xml"),
}
breathe_default_project = "trx-cpp"

primary_domain = "cpp"
highlight_language = "cpp"

autosectionlabel_prefix_document = True

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "API Reference",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": False,
}
