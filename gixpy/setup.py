from setuptools import setup, Extension
from pathlib import Path
import numpy as np

gp_module = Extension(
    "gixpy.transform",
    sources=["gixpy/transform.c"],
    include_dirs=[
        np.get_include(),
        ],
    library_dirs=["\\root\\project"],
    language="c",
)

setup(
    name='gixpy',
    version="5.6",
    description="Python package to quickly transform GIWAXS images using C",
    author="Teddy Tortorici",
    long_description=Path('README.md').read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author_email='edward.tortorici@colorado.edu',
    url='https://github.com/ttortorici/gixpy',
    packages=['gixpy'],
    license='GPL-3.0',
    ext_modules=[gp_module],
    install_requires=["numpy", "pyFAI", "fabio"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)