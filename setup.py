"""
setup.py

After running: pip install -e .
The 'fracture' command is available globally.

Usage:
    fracture demo
    fracture run --pipeline-id trade_batch
    fracture status
"""

from setuptools import setup, find_packages

setup(
    name='fracture',
    version='0.1.0',
    description='Process conformance linter for data pipelines',
    author='Siddhant Nayak',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'pm4py',
        'pydantic>=2.0',
        'pyyaml',
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn',
        'pyarrow',
    ],
    entry_points={
        'console_scripts': [
            'fracture=fracture.cli:main',
        ],
    },
)
