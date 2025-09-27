from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ultrasound-roi",
    version="1.0.0",
    author="Kamlin Kambaram",
    author_email="kamlin.ekambaram@gmail.com",
    description="Automated Region of Interest Segmentation and De-identification for Echocardiogram Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation",
    project_urls={
        "Bug Reports": "https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation/issues",
        "Source": "https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation",
        "Documentation": "https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation#readme",
        "Paper": "https://joss.theoj.org/papers/[to-be-assigned]",
    },
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    packages=find_packages(),
    py_modules=["unet_inference"],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultrasound-roi=unet_inference:main",
        ],
    },
    keywords="unet, ultrasound, medical imaging, segmentation, deep learning, echo-vivit",
    zip_safe=False,
)
