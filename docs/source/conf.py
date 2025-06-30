import sys
from pathlib import Path
from datetime import datetime
import toml
from sphinx_gallery.sorting import FileNameSortKey

# Setup paths
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))

# Project Information from pyproject.toml
pyproject_data = toml.load(base_dir / "pyproject.toml")
project_info = pyproject_data["project"]

project = project_info["name"]
author = ", ".join([f"{a.get('name', '')} ({a.get('email', '')})" for a in project_info.get("authors", [])])
release = version = project_info["version"]
copyright = f"2025 - {datetime.now().year}, n-squared lab, FAU Erlangen-Nürnberg, Germany"

# Import the main package
import myogen

# Sphinx Configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.doctest",
    "myst_parser",
]

# MyST-Parser configuration
myst_enable_extensions = [
    "attrs_inline",
    "attrs_block", 
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
myst_admonition_enable = True
myst_url_schemes = ["http", "https", "mailto", "ftp"]
myst_ref_domains = ["std"]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_inherit_docstrings = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# Autosummary configuration
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# General configuration
templates_path = ["templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store"]

pygments_dark_style = "monokai"  

# HTML theme configuration
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": f"https://github.com/NSquaredLab/{project}",
    "navbar_start": ["navbar-logo", "navbar-version.html", "header-text.html"],
    "show_prev_next": False,
    "navbar_align": "left",
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    # Pygments (syntax highlighting) configuration
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_static_path = ["_static"]
html_logo = "_static/myogen_logo.png"
html_css_files = ["custom.css"]
html_title = f"{project} {version} Documentation"

# HTML context
html_context = {
    "AUTHOR": author,
    "VERSION": version,
    "DESCRIPTION": project_info.get("description", ""),
    "github_user": "NSquaredLab",  # Update with your GitHub username
    "github_repo": project,
    "github_version": "main",
    "doc_path": "docs",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": str(base_dir / "examples"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py",
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "show_memory": False,
    "plot_gallery": True,
    "download_all_examples": False,
    "first_notebook_cell": "%matplotlib inline",
}

# Warning suppressions
suppress_warnings = [
    "config.cache",
    "ref.citation",
]

def setup(app):
    """Setup function for custom configurations."""
    pass