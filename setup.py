#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    author="Julian Klug",
    author_email='tensu.wave@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=(
        "Tools for the prepocessing of 3D medical images"
    ),
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords=['geneva_stroke_preprocessing', 'perfusion', 'CT', 'pCT', 'stroke'],
    name='gsprep',
    packages = find_packages(exclude=('examples', 'illustrations')),
    setup_requires=[],
    url='https://github.com/MonsieurWave/Geneva-Stroke-Preprocessing',
    version='0.1.0',
)