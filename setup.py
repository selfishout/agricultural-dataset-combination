#!/usr/bin/env python3
"""
Setup script for Agricultural Dataset Combination Project.

This project combines multiple agricultural datasets into a unified format
suitable for Weakly Supervised Semantic Segmentation (WSSS) applications.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Agricultural Dataset Combination Project"

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Project metadata
PROJECT_NAME = "agricultural-dataset-combination"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "A comprehensive project for combining multiple agricultural datasets into a unified format suitable for Weakly Supervised Semantic Segmentation (WSSS) applications."
PROJECT_LONG_DESCRIPTION = read_readme()
PROJECT_AUTHOR = "Ali Torabi"
PROJECT_AUTHOR_EMAIL = "ali.torabi@example.com"
PROJECT_URL = "https://github.com/selfishout/agricultural-dataset-combination"
PROJECT_LICENSE = "MIT"
PROJECT_CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Agricultural",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Testing",
    "Topic :: Software Development :: Documentation",
]

# Package configuration
PACKAGES = find_packages(where="src")
PACKAGE_DIR = {"": "src"}

# Entry points
ENTRY_POINTS = {
    "console_scripts": [
        "combine-datasets=scripts.combine_datasets:main",
        "setup-datasets=scripts.setup_datasets:main",
        "validate-combination=scripts.validate_combination:main",
    ],
}

# Setup configuration
setup(
    name=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=PROJECT_DESCRIPTION,
    long_description=PROJECT_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=PROJECT_AUTHOR,
    author_email=PROJECT_AUTHOR_EMAIL,
    url=PROJECT_URL,
    license=PROJECT_LICENSE,
    classifiers=PROJECT_CLASSIFIERS,
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "agriculture",
        "dataset",
        "computer-vision",
        "image-segmentation",
        "machine-learning",
        "deep-learning",
        "phenotyping",
        "plant-analysis",
        "weed-detection",
        "crop-monitoring",
        "precision-agriculture",
        "wsss",
        "weakly-supervised",
        "semantic-segmentation",
        "pytorch",
        "opencv",
        "albumentations",
        "data-processing",
        "data-augmentation",
        "image-preprocessing",
    ],
    project_urls={
        "Bug Reports": f"{PROJECT_URL}/issues",
        "Source": PROJECT_URL,
        "Documentation": f"{PROJECT_URL}/blob/main/README.md",
        "Changelog": f"{PROJECT_URL}/blob/main/CHANGELOG.md",
        "Contributing": f"{PROJECT_URL}/blob/main/CONTRIBUTING.md",
        "License": f"{PROJECT_URL}/blob/main/LICENSE",
    },
    # Additional metadata
    maintainer=PROJECT_AUTHOR,
    maintainer_email=PROJECT_AUTHOR_EMAIL,
    platforms=["any"],
    requires_python=">=3.8",
    # Development dependencies (optional)
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipykernel>=6.0.0",
        ],
    },
)

if __name__ == "__main__":
    setup()
