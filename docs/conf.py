import os

project = "trx-cpp"
author = "trx-cpp contributors"

extensions = [
    "myst_parser",
    "breathe",
    "exhale",
    "sphinx.ext.autosectionlabel",
]

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tee-ar-ex/trx-cpp",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_toc_level": 2,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

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
