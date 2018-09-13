.. hilbert documentation master file, created by
   sphinx-quickstart on Wed Sep 12 23:34:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hilbert --- simple embedding framework for deep learning.
========================================================

``Embedings``
~~~~~~~~~~~~~

Typical usage:

.. code-block:: python

    my_embeddings = hilbert.embeddings.Embeddings(U, V, dictionary)

The ``Embeddings`` class let's you easily manipulate embeddings.  You can read
or save them to disk, get the embedding for a given word, normalize them, or
find the embedding most similar to a given one.

All the word vectors are stored in a 2D tensor ``my_embeddings.V``, with one
vector per row.  If covectors were included, they are similarly structured and
are at ``my_embeddings.W``.

If you just want to access the underlying tensors or dictionary, just do:

.. code-block:: python

    V, W, dictionary = my_embeddings


Usually, you'll obtain embeddings in one of these ways:

(1) Generating them randomly:

    ... code-block:: python

        hilbert.embeddings.random(d=300, vocab=100000)

(2) Training them:

    .. code-block:: python

        >>> # supposing you have a hilbert.embedder.HilbertEmbedder...
        >>> while not my_embedder.converged:
        ...    my_embedder.cycle()    
        >>> my_embeddings = my_embedder.get_embeddings()

    (after which you would normally save them by doing
    ``my_embeddings.save('path-to-my-embeddings'.``)

(3) Reading saved embeddings from disk:

    .. code-block:: python

        >>> embeddings = h.embedings.Embeddings.load('path-to-my-embeddings')

(4) Or making them manually from some torch tensors or numpy arrays:

    .. code-block:: python

        >>> import torch
        >>> dimensions, vocab = 300, 5000
        >>> V = torch.rand(dimensions, vocab)
        >>> W = torch.rand(vocab, dimensions)
        >>> embeddings = h.embeddings.Embeddings(V, W)

    Notice the shape of the embeddings.  `V` has word vectors as its columns,
    while 'W' has its word vectors as rows.  There's a good chance I'll change
    this so that `V` and `W` would have the same shape, but I find it less
    appealing when written as math.

    .. todo::

        Make it so that the shape of `V` and `W` are the same.


Embeddings can use a dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you provide a dictionary when you make them, then you can access the
embeddings for given word by its name:

.. code-block:: python

    # you can get a toy dictionary for testing like so...
    >>> my_embeddings = hilbert.embeddings.random(300, 5000, dictionary)
    >>> my_embeddings['dog']
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])

The dictionary should be a ``hilbert.dictionary.Dictionary`` instance.
A 5000-word dictionary is available for testing purposees by doing
``hilbert.test.get_test_dictionary()`` (you will need to explicitly import
``hilbert.test``).


Accessing embeddings of specific words.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we saw, you can access the embedding for a given word either using
its name or its index in the dictionary.

.. code-block:: python

    >>> embeddings['dog']
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])
    >>> embeddings[3170]
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])
    >>> dictionary.get_id('dog') === 3170
    True

.. py:module:: hilbert

.. autoclass:: embeddings.Embeddings
    :member-order: bysource
    :members:







