from setuptools import setup, Extension, find_packages
import numpy as np

with open ("README.md", "r") as f:
    long_description = f.read()

c_module = Extension(
    name="_gixpy",
    sources=["gixpy\\gixpy.c"],
    include_dirs=[np.get_include()],
    language="c",
)

# See pyproject.toml for metadata
setup(
    name="gixpy",
    packages=find_packages(include=["gixpy", "gixpy.*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[c_module],
)
