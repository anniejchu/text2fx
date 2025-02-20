from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="text2fx",
    version="0.0.1",
    classifiers=[
    ],
    description="text-guided audio FX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Annie Chu, Patrick O'Reilly",
    author_email="anniechu@u.northwestern.edu",
    python_requires=">=3.9",
    url="https://github.com/anniejchu/text2fx",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
)