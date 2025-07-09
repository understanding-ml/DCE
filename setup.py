from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="distributional-counterfactual-explanation",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    author="Lei You, Lele Cao, Yikai Gu",
    author_email="leiyo@dtu.dk, lele.cao@eqtpartners.com",
    description="Distributional Counterfactual Explanation with Optimal Transport for Non-differentiable Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youlei202/distributional-counterfactual-explanation",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning, explainable-ai, counterfactual-explanation, optimal-transport, xai",
    project_urls={
        "Bug Reports": "https://github.com/youlei202/distributional-counterfactual-explanation/issues",
        "Source": "https://github.com/youlei202/distributional-counterfactual-explanation",
        "Documentation": "https://github.com/youlei202/distributional-counterfactual-explanation#readme",
    },
    entry_points={
        "console_scripts": [
            "dce-experiment=experiments.cardio_mlp_unified:main",
        ],
    },
)