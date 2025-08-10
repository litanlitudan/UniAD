"""Setup script for UniAD Model Analyzer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="uniad-model-analyzer",
    version="0.1.0",
    author="UniAD Model Analyzer Team",
    author_email="",
    description="A comprehensive PyTorch model analysis tool for UniAD autonomous driving model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opendrivelab/UniAD",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
        "tabulate>=0.8.9",
        "jinja2>=3.0.0",
        "plotly>=5.0.0",
        "mmcv-full>=1.4.0",
        "mmdet>=2.19.0",
        "mmsegmentation>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "isort>=5.9",
        ],
        "visualization": [
            "seaborn>=0.11.0",
            "bokeh>=2.3.0",
        ],
        "export": [
            "tensorboard>=2.6.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "uniad-analyze=uniad_model_analyzer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "uniad_model_analyzer": ["templates/*.html", "templates/*.css"],
    },
    zip_safe=False,
)