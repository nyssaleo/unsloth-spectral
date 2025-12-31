"""
Setup script for unsloth-spectral

Installation:
    pip install -e .           # Editable install (development)
    pip install .              # Standard install
    pip install unsloth-spectral  # From PyPI (future)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = requirements_file.read_text().strip().split("\n") if requirements_file.exists() else []

setup(
    name="unsloth-spectral",
    version="0.1.0",
    author="Ankit Prajapati",
    author_email="your.email@example.com",  # Update this
    description="Holographic Spectral Compression for LLM KV Caches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/unsloth-spectral",  # Update this
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
        ],
        "colab": [
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        ],
    },
    entry_points={
        "console_scripts": [
            "unsloth-spectral-test=unsloth_spectral.cli:test_command",  # Future CLI
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

