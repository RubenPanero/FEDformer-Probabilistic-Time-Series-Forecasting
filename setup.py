"""
Setup script for Vanguard-FEDformer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="vanguard-fedformer",
    version="0.1.0",
    author="Vanguard Team",
    author_email="your.email@example.com",
    description="Advanced Probabilistic Time Series Forecasting with FEDformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting/issues",
        "Source": "https://github.com/yourusername/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting",
        "Documentation": "https://github.com/yourusername/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vanguard-train=vanguard_fedformer.scripts.train:main",
            "vanguard-evaluate=vanguard_fedformer.scripts.evaluate:main",
            "vanguard-demo=vanguard_fedformer.scripts.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vanguard_fedformer": [
            "configs/*.yaml",
            "data/sample/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "time-series",
        "forecasting",
        "transformer",
        "fedformer",
        "probabilistic",
        "deep-learning",
        "pytorch",
        "finance",
        "regime-detection",
        "risk-analysis",
    ],
)