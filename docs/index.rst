.. hilbert documentation master file, created by
   sphinx-quickstart on Wed Sep 12 23:34:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hilbert --- simple embedder for deep learning.
==============================================

Embeddings
~~~~~~~~~~

Typical usage:

.. code-block:: python

    my_embeddings = hilbert.embeddings.Embeddings(V, W, dictionary)

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

    .. code-block:: python

        hilbert.embeddings.random(vocab=100000, d=300)

(2) Training them:

    .. code-block:: python

        >>> # supposing you have a hilbert.embedder.HilbertEmbedder...
        >>> while not my_embedder.converged:
        ...    my_embedder.cycle()    
        >>> my_embeddings = my_embedder.get_embeddings()

    (And then save them:  ``my_embeddings.save('path-to-my-embeddings'``.)

(3) Reading saved embeddings from disk:

    .. code-block:: python

        >>> embeddings = h.embeddings.Embeddings.load('path-to-my-embeddings')

(4) Or making them manually from some torch tensors or numpy arrays:

    .. code-block:: python

        >>> import torch
        >>> vocab, dimensions = 5000, 300
        >>> V = torch.rand(vocab, dimensions)
        >>> W = torch.rand(vocab, dimensions)
        >>> embeddings = h.embeddings.Embeddings(V, W)

    Notice that the vector and covector arrays should have one vector per row.


Embeddings can use a dictionary.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dictionaries map tokens to integer IDs, which themselves correspond to the
index of that token's embedding within ``V`` and ``W``.  When embeddings have an
associated dictionary, you can access vectors for given words by name:

.. code-block:: python

    >>> # you can get a toy dictionary for testing like so...
    >>> import hilbert.test
    >>> dictionary = hilbert.test.get_test_dictionary()
    >>>
    >>> my_embeddings = hilbert.embeddings.random(300, 5000, dictionary)
    >>> my_embeddings['dog']
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])
    >>>
    >>> # Note the equivalence
    >>> all(my_embeddings['dog'] == my_embeddings[dictionary.get_id('dog')])
    True

The dictionary should be a ``hilbert.dictionary.Dictionary`` instance.
A 5000-word dictionary is available for testing purposees by doing
``hilbert.test.get_test_dictionary()`` (you will need to explicitly import
``hilbert.test``).


Accessing embeddings of specific words.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can always access a vector by index.

.. code-block:: python

    >>> embeddings[3170]
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])

Any slicey stuff will be sent right through to the underlying tensors / arrays:

.. code-block:: python
    
    >>> embeddings[1:6:2, :3]
    tensor([[0.6240, 0.2314, 0.4231],
            [0.7956, 0.7815, 0.4875],
            [0.7281, 0.8238, 0.9222]])

In most cases, vectors, as opposed to covectors, are desired.  To access covectors, use the `get_covec()` method considered:

.. code-block:: python
    
    >>> embeddings.get_covec(3170)
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])
    >>> embeddings.get_covec('dog')
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])

(or just slice into the `W` attribute.)


Reference for ``hilbert.embeddings.Embeddings``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:module:: hilbert

.. autoclass:: embeddings.Embeddings
    :member-order: bysource
    :members:


Reference for ``hilbert.embeddings.random``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: embeddings.random


