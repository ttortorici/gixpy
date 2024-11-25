from setuptools import setup, Extension, find_packages
import numpy as np

with open ("README.md", "r") as f:
    long_description = f.read()

c_module = Extension(
    name="gixpy_c",
    sources=["source\\gixpy.c"],
    include_dirs=[np.get_include()],
    language="c",
)

setup(
    name='gixpy',
    version="2.0",
    packages=find_packages(include=["source", "source.*"]),
    description="Python package to quickly transform images from grazing incidence X-ray experiments using C",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='edward.tortorici@colorado.edu',
    license='GPL',
    ext_modules=[c_module],
    install_requires=[
        "numpy>=1.7",
        "pyFAI",
        "fabio",
        "matplotlib",
    ], # numpy 2.1.3 works, and future versions should work
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: GPL License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)