from setuptools import setup
import re

with open('README.md') as f:
    long_description = f.read()

setup(
    name='sphinx-autodoc-construct',
    version='0.1.0',
    description='A Sphinx extension to automatically document your Contruct structs.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/maxdup/sphinx-autodoc-construct',
    license='gpl-3.0',
    author='Maxime Dupuis',
    author_email='mdupuis@hotmail.ca',
    packages=['sphinx_autodoc_construct'],
    package_data={
        'sphinx_autodoc_construct': [
            "_static/sphinx-autodoc-construct.css",
        ],
    },
    install_requires=[
        'construct',
        'executing',
        'mock'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Documentation',
        'Topic :: Software Development',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Framework :: Sphinx :: Extension',
    ],
)
