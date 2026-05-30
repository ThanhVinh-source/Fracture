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

        # Analytical extensions used by changepoint detection and graph analysis.
        'ruptures',
        'graphviz',
        'networkx',

        # Visualization/dashboard runtime for Phase 2+.
        # Keeping these in setup.py makes `pip install -e .` usable for demos.
        'streamlit',
        'plotly',
        'matplotlib',
        'seaborn',

    ],
    entry_points={
        'console_scripts': [
            'fracture=fracture.cli:main',
        ],
    },
)
