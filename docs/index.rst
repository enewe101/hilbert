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

The ``Embeddings`` class tries to be useful, but if you just want to access the underlying tensors or dictionary, just do:

.. code-block:: python

    V, W, dictionary = my_embeddings


Usually, you'll obtain embeddings in one of these ways:

(1) Generating them randomly, e.g. by calling ``hilbert.embeddings.random(d=300,
    vocab=100000)``

(2) Training them.  E.g.

    .. code-block:: python

        # suppose you have a hilbert.embedder.HilbertEmbedder
        >>> while not my_embedder.converged:
        ...    my_embedder.cycle()    
        >>> my_embeddings = my_embedder.get_embeddings()


    (after which you would normally save them by doing
    ``my_embeddings.save('path-to-my-embeddings'.``)

(2) Reading saved embeddings from disk:

    .. code-block:: python

        >>> embeddings = h.embedings.Embeddings.load('path-to-my-embeddings')

(3) Or making them manually from some torch tensors or numpy arrays:

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
    >>> import hilbert.test
    >>> dictionary = hilbert.test.get_test_dictionary()
    >>> my_embeddings = hilbert.embeddings.random(300, 5000, dictionary)
    >>> my_embeddings['dog']
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])


Accessing embeddings of specific words.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we saw, you can access the embedding for a given word by just indexing with
it's name.  You can also index by providing an `int`


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


.. py:module:: hilbert

.. autoclass:: embeddings.Embeddings
    :member-order: bysource
    :members:

        .. automethod:: save


.. py:class:: hilbert.embeddings.Embeddings

        .. py:method:: yoyo()




