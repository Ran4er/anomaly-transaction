"""
Setup script for transaction anomaly detection package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="transaction-anomaly-detection",
    version="1.0.0",
    author="Ran4er",
    author_email="khromovdaniel23@gmail.com",
    description="Production-ready anomaly detection for financial transactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ran4er/anomaly-transaction",
    project_urls={
        "Bug Tracker": "https://github.com/Ran4er/anomaly-transaction/issues",
        "Documentation": "https://github.com/Ran4er/anomaly-transaction/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "anomaly-train=scripts.train_model:main",
            "anomaly-generate-data=scripts.generate_data:main",
        ],
    },
)