from setuptools import setup, find_packages

setup(
    name="inside_traders",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "matplotlib>=3.7"],
    extras_require={"dev": ["pytest>=7.4", "networkx>=3.1"]},
)
