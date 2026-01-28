import os

project = "TRX-cpp"
author = "TRX-cpp contributors"
release = "0.0.0"

extensions = [
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

source_suffix = ".rst"
master_doc = "index"

breathe_projects = {
    "trx-cpp": os.path.join(os.path.dirname(__file__), "_build", "xml"),
}
breathe_default_project = "trx-cpp"

primary_domain = "cpp"
highlight_language = "cpp"

