from setuptools import setup, Extension

module = Extension(
'symnmfmodule',
sources=['symnmfmodule.c', 'symnmf.c'],
)

setup(
name='symnmfmodule',
version='1.0',
description='Symmetric NMF C extension module',
ext_modules=[module],
)