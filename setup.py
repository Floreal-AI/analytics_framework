# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
"""
setup.py – Package definition for `conversion_subnet`
=====================================================

This script builds and installs the Real‑time Conversation Analytics subnet
for Bittensor.  It relies on a *single* source of truth for the version
(`conversion_subnet/__init__.py`) to avoid drift between code and wheel
metadata.

Typical usage
-------------
$ python -m build            # build sdist + wheel into ./dist
$ pip install -e .[dev]      # editable install with dev extras
"""

from __future__ import annotations

import re
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()
INIT_PATH = ROOT_DIR / "conversion_subnet" / "__init__.py"

_spec = spec_from_file_location("conversion_subnet", INIT_PATH)
pkg = module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(pkg)  # type: ignore[attr-defined]
VERSION: str = pkg.__version__

README_MD = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
# Strip HTML comments that hide badge markup from GitHub
LONG_DESCRIPTION = re.sub(r"<!--.*?-->", "", README_MD, flags=re.DOTALL)

INSTALL_REQUIRES = [
    "bittensor>=6.0.0",
    "torch>=2.0.0",
    "pandas>=1.5.0",
    "faker>=18.0.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.2.0",
    "numpy>=1.23.0",
    "loguru>=0.6.0",
    "password-strength==0.0.3.post2",
]

EXTRAS_REQUIRE = {
    "dev": [
        "ruff",
        "bump2version",
    ],
    "test": [
        "pytest",
        "pytest-cov",
        "tox",
    ],
    "docs": [
        "mkdocs",
        "mkdocs-material",
    ],
}

ENTRY_POINTS = {
    "console_scripts": [
        # Thin wrappers located in neurons/; each wrapper calls a CLI function
        "miner-conversion     = neurons.miner:cli",
        "validator-conversion = neurons.validator:cli",
    ],
}

PACKAGES = find_namespace_packages(
    include=[
        "conversion_subnet",
        "conversion_subnet.*",
        "neurons",
        "neurons.*",
    ]
)
setup(
    name="conversion_subnet",
    version=VERSION,
    description="Real-time Conversation Analytics Bittensor subnet",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Floreal AI",
    author_email="info@floreal.ai",
    url="https://github.com/Floreal-AI/analytics_framework",
    project_urls={
        "Source": "https://github.com/Floreal-AI/analytics_framework",
        "Issue Tracker": "https://github.com/Floreal-AI/analytics_framework/issues",
    },
    license="MIT",
    python_requires=">=3.8",
    packages=PACKAGES,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,  # we may bundle weight files / data assets
)

