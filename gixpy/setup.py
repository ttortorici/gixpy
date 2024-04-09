from setuptools import setup, Extension

gp_module = Extension("gixpy",
                      sources=["gixpy.c"],
                      include_dirs=[
                          'include', 
                          '..\\gixpy', 
                          'C:\\Users\\Teddy\\AppData\\Roaming\\Python\\Python39\\site-packages\\numpy\\core\\include'],
                      library_dirs=["\\root\\project"],
                      language="c",
                      )

setup(
    name='gixpy',
    version="1.0",
    description="Python package to quickly transform GIWAXS images using C",
    ext_modules=[gp_module],
    install_requires=["numpy"],
)