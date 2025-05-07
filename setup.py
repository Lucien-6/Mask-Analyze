#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安装配置：提供通过pip安装的支持
"""
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="mask-analyze",
    version="1.0.0",
    description="掩膜数据分析工具：用于处理和分析掩膜图像序列的工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="YourName",
    author_email="your.email@example.com",
    url="https://github.com/yourname/mask-analyze",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'mask-analyze=main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.6",
) 