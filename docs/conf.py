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
html_logo = "_static/trx_logo.png"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tee-ar-ex/trx-cpp",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "image_light": "_static/trx_logo.png",
        "image_dark": "_static/trx_logo.png",
        "alt_text": "TRX",
        "link": "https://tee-ar-ex.github.io",
    },
    "show_toc_level": 2,
    "navigation_depth": 4,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
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
