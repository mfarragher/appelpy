import setuptools

with open("docs/index.md", "r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

PROJECT_URLS = {"Documentation": "https://appelpy.readthedocs.io/",
                "Source": "https://github.com/mfarragher/appelpy"}
INSTALL_REQUIRES = ["pandas>=0.24", "jinja2",
                    "scipy",
                    "numpy>=1.16",
                    "statsmodels>=0.9", "patsy",
                    "seaborn>=0.9",
                    "matplotlib>=3"]
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Office/Business :: Financial",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
]

setuptools.setup(
    name="appelpy",
    version="0.4.2",
    author="Mark Farragher",
    description="Applied Econometrics Library for Python",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=["econometrics", "regression",
              "statistics", "economics", "models"],
    url="https://github.com/mfarragher/appelpy",
    project_urls=PROJECT_URLS,
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    license="BSD",
    classifiers=CLASSIFIERS
)
