"""
VANGUARD: Variational Anatomical Neural Generator for Unified and Adaptive Reconstruction in Diverse imaging
Setup configuration for Python package installation
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(f"requirements/{filename}", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-r")]

setup(
    name="vanguard-mri-harmonization",
    version="0.1.0",
    author="Daniel Barreras",
    author_email="daniel.barreras@research.org",
    description="Bayesian deep learning for cross-site MRI harmonization with uncertainty quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielBrMz/vanguard-mri-harmonization",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("base.txt"),
    extras_require={
        "training": read_requirements("training.txt"),
        "evaluation": read_requirements("evaluation.txt"),
        "development": read_requirements("development.txt"),
        "all": read_requirements("base.txt") + read_requirements("training.txt") + read_requirements("evaluation.txt"),
    },
    entry_points={
        "console_scripts": [
            "vanguard-train=vanguard.training.train:main",
            "vanguard-evaluate=vanguard.evaluation.evaluate:main",
            "vanguard-preprocess=vanguard.data.preprocess:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vanguard": ["configs/*.yaml", "data/templates/*"],
    },
    zip_safe=False,
)
