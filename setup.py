import setuptools

with open("README.md", "r", encoding="utf8", errors="ignore") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FEAutils",
    version="0.0.1",
    author="Future Energy Associates",
    author_email="hello@futureenergy.associates",
    description="Helper utilities useful in energy data science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/future-energy-associates/FEA-helpers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.6",
)
