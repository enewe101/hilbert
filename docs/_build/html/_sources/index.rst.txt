.. hilbert documentation master file, created by
   sphinx-quickstart on Wed Sep 12 23:34:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``hilbert`` --- a simple embedding framework for deep learning
==============================================================

``Embedings``
~~~~~~~~~~~~~

The ``hilbert.embedings.Embeddings`` class allows you to save, load, and do
some basic manipulations of embeddings.  At its core, the ``Embeddings`` class
owns a set of (word-)vector and (context-)covector embeddings, which are
contained in two torch tensors (``torch.Tensor(dtype=torch.float32)``), along
with a dictionary that maps between words and vector indices.

If you just want the class to get out of your way, and give you access to those
underlying data, do this: ``vectors, covectors, dictionary = embeddings``.

However, the ``Embeddings`` class provides a couple conveniences.  To begin,
you can get some random embeddings by doing:

.. code-block:: python

    >>> import hilbert as h
    >>> embeddings = h.embeddings.random(d=300, vocab=5000)

Embeddings can optionally associate a dictionary, which makes it easy to 
access the embeddings that correspond to particular words.  Here we're loading
a small dictionary used in testing, but normally you would use a dictionary
built from accumulating cooccurrence statistics:

.. code-block:: python

    >>> import hilbert.test as t
    >>> dictionary = t.get_test_dictionary()
    >>> embeddings = h.embeddings.random(300, 5000, dictionary=dictionary)
    >>> embeddings['dog']
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])

Normally, you would either (1) get embeddings from an embedder, (2) read
embeddings previously stored on disk, or (3) make an instance by passing in
raw torch tensors or numpy arrays.

(1) Get embeddings from an embedder:

.. code-block:: python

    >>> # suppose you have an embedder
    >>> embedder
    <class 'hilbert.torch_embedder.TorchHilbertEmbedder'>
    >>> embeddings = embedder.get_embeddings()

(2) Read previously stored embeddings:

.. code-block:: python

    >>> embeddings = h.embedings.Embeddings.load('my-saved-embeddings')

(3) Make an instance by passing in raw torch tensors or numpy arrays

.. code-block:: python

    >>> import torch
    >>> dimensions, vocab = 300, 5000
    >>> V = torch.rand(dimensions, vocab)
    >>> W = torch.rand(vocab, dimensions)
    >>> embeddings = h.embeddings.Embeddings(V, W, embeddings)




