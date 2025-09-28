#!/usr/bin/env python
"""Setup script for UNET Ultrasound ROI Segmentation package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "U-Net based ultrasound ROI segmentation tool"

# Read requirements
def read_requirements():
    """Read requirements.txt for dependencies."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f.readlines() 
                          if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name="unet-ultrasound-roi",
    version="1.0.0",
    author="Kamlin Ekambaram",
    author_email="kamlinekambaram@gmail.com",  # Update with actual email
    description="U-Net based deep learning model for ultrasound ROI segmentation and deidentification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kamlinekambaram/UNET-Ultrasound-ROI-Segmentation",  # Update with actual URL
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "unet-roi=unet_roi.cli:main",
            "unet-roi-train=unet_roi.cli:train_cli",
            "unet-roi-predict=unet_roi.cli:predict_cli",
            "unet-roi-evaluate=unet_roi.cli:evaluate_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "unet_roi": ["models/*.keras", "data/sample/*"],
    },
    zip_safe=False,
    keywords="deep-learning machine-learning medical-imaging ultrasound segmentation u-net tensorflow keras",
)