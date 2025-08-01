from setuptools import setup, find_packages

setup(
    name="XAI4EO",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "optuna",
        "opencv-python",
        "xgboost",
        "lightgbm",
        "h2o>=3.36.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for automated machine learning on hyperspectral and multispectral imaging data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/XAI4EO",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)