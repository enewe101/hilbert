import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hilbert",
    version="0.0.1",
    author="Edward Newell",
    author_email="edward.newell@gmail.com",
    description=(
	"Word embeddings, hilbert embedder, glove, word2vec, skip gram"
	"fasttext, nlp, deep learning."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enewe101/hilbert-research",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','scipy']
)
