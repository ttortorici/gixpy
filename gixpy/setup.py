from setuptools import setup, Extension
from pathlib import Path
import numpy as np

gp_module = Extension(
    "gixpy",
    sources=["gixpy.c"],
    include_dirs=[
        np.get_include(),
        #Path.cwd().parent / "include",
        #'..\\include', 
        #'..\\gixpy', 
        #'C:\\Users\\Teddy\\AppData\\Roaming\\Python\\Python39\\site-packages\\numpy\\core\\include'
        ],
    library_dirs=["\\root\\project"],
    language="c",
)

setup(
    name='gixpy',
    version="2.0",
    description="Python package to quickly transform GIWAXS images using C",
    long_description="Visit github.com/etortorici/gixpy for details",
    author_email='edward.tortorici@colorado.edu',
    url='https://github.com/etortorici/gixpy',
    license='GPL',
    ext_modules=[gp_module],
    install_requires=["numpy"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GPL License',
        'Operating System :: Microsoft :: Windows',
    ],
    python_requires='>=3.9',
)