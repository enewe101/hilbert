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

Embeddings often come from one of these places:
(1) Generated randomly, e.g. by calling ``hilbert.embeddings.random(d=300,
    vocab=100000)``

(2) By training them.  E.g.

.. code-block:: python

    # suppose you have a hilbert.embedder.HilbertEmbedder
    >>> while not my_embedder.converged:
    ...    my_embedder.cycle()    
    >>> my_embeddings = my_embedder.get_embeddings()


    (after which you would normally save them by doing
    ``my_embeddings.save('path-to-my-embeddings'.``)

(2) Read previously embeddings:

.. code-block:: python

    >>> embeddings = h.embedings.Embeddings.load('path-to-my-embeddings')

(3) Or, if you happen to have raw numpy arrays or torch tensors, you can use
    those to make the embedding:

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


.. py:module:: hilbert

.. autoclass:: embeddings.Embeddings
    :member-order: bysource
    :members:

        .. automethod:: save


.. py:class:: hilbert.embeddings.Embeddings

        .. py:method:: yoyo()




