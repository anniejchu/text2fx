from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="text2fx",
    version="0.0.1",
    classifiers=[
    ],
    description="Generative Music Modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="???",
    author_email="??",
    url="????",
    license="MIT",
    packages=find_packages(),
    install_requires=[
    ],
)