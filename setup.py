from setuptools import setup, find_packages

setup(
    name='totalmrvi',
    version='0.1.0',
    description='Package built in scvi-tools for TOTALMRVI, TOTALMRVAE and its components',
    author='Sreyas Adiraju',
    author_email='sreyas.adiraju@columbia.edu',
    packages=find_packages(), # Automatically find 'totalmrvi'
    install_requires=[
        'torch>=2.6.0', # Specify your torch version dependency
        'numpy',
        'scvi-tools>=1.3.0' # Specify scvi-tools version if applicable
    ],
    python_requires='>=3.11', # Specify your Python version
)