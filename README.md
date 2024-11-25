# Installation

GixPy is distributed on PyPI and can be installed using pip:

```
pip install gixpy
```

It can be built from source by cloning https://github.com/ttortorici/gixpy.git and using Python's `build` tool (don't forget to `cd` into the repository):

```
pip install -U setuptools build
python -m build --wheel
pip install dist/*.whl
```

where `*` should be replaced with the actual wheel's filename, and `\` should be used instead of `/` on Windows.



