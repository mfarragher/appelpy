import setuptools

with open("README.rst", "r", encoding="utf8") as f:
    long_description = f.read()

setuptools.setup(
    name="appelpy",
    version="0.4.0",
    author="Mark Farragher",
    description="Applied Econometrics Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["econometrics", "regression",
              "statistics", "economics", "models"],
    url="https://github.com/mfarragher/appelpy",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    install_requires=["pandas>=0.24", "jinja2",
                      "scipy",
                      "numpy>=1.16",
                      "statsmodels>=0.9", "patsy",
                      "seaborn>=0.9",
                      "matplotlib>=3"],
    license="BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
