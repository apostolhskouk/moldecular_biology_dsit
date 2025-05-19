from setuptools import setup, find_packages

setup(
    name="ChemFlow",  # The name for pip install and import
    version="0.1.0",
    # Find packages *inside* the 'src' directory
    packages=find_packages(where="src"),
    # Tell setuptools that the package root ('') corresponds to the 'src' directory
    package_dir={"": "src"},
    # Add other metadata like author, description, install_requires if needed
)