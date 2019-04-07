import setuptools

with open("README.rst", "r", encoding="utf8") as f:
    long_description = f.read()

setuptools.setup(
    name="appelpy",
    version="0.0.2",
    author="Mark Farragher",
    description="Applied Econometrics Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["econometrics", "regression",
              "statistics", "economics", "models"],
    url="https://github.com/mfarragher/appelpy",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    install_requires=["pandas>=0.24",
                      "scipy",
                      "numpy",
                      "statsmodels>=0.8", "patsy",
                      "seaborn",
                      "matplotlib"],
    license="BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
