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

import re
import os
import codecs
import pathlib
from os import path
from io import open
from setuptools import setup, find_packages
from pkg_resources import parse_requirements


def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []

        for req in requirements:
            # For git or other VCS links
            if req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    # You may decide to raise an exception here,
                    # if you want to ensure every VCS link has an #egg=<package_name> at the end
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements


requirements = read_requirements("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "conversion_subnet/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)


setup(
    name="conversion_subnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "bittensor>=6.12.3",
        "numpy>=1.21.0",
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": ["scikit-learn>=1.0.0"],
    },
    author="Floreal AI",
    author_email="bg@floreal.ai",
    description="A Bittensor subnet for predicting quote delivery time and acceptance rate",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Floreal-AI/analytics_framework",
    license="MIT",
    python_requires=">=3.8,<3.12",
)
