import setuptools

with open("pip-long-description.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hilbert",
    version="0.0.1",
    author="Edward Newell",
    author_email="edward.newell@gmail.com",
    description=("Hilbert---The Canonical Embedding Library"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enewe101/hilbert",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'nose2', 'numpy', 'scipy', 'matplotlib', 'pytorch-categorical']
)

print(
    'Install the appropriate version of pytorch.  Go to '
    'https://pytorch.org/'
)
