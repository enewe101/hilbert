.. hilbert documentation master file, created by
   sphinx-quickstart on Wed Sep 12 23:34:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hilbert --- simple embedder for deep learning.
==============================================

Installation
~~~~~~~~~~~~
Hilbert uses Pytorch; you'll need that installed first.
Then do

.. code-block:: python

    pip install hilbert



Make Embeddings
~~~~~~~~~~~~~~~

You'll need to run two scripts to make embeddings.  The first extracts bigram
statistics from your corpus, and stores them in a format that is efficient for
training:

.. code-block:: python
    
    python extract.py etc...

The second one uses the bigram statistics to train embeddings
according to your choice of model, and writes the embeddings to disk.  Once
you've extracted bigram data once, you can use it to train many different
models.

Read on for more details about the two commands.


Extract Corpus Statistics:
^^^^^^^^^^^^^^^^^^^^^^^^^^
The corpus file needs to be space-tokenized, and each newline creates a new
*cooccurrence context*.  Usually, this means you'll format the corpus to have
each *document* on its own line, whatever "document" means to you.  Whith that
formatting, only words *within* a document can be considered as cooccurring (if
they are less than ``--window-size`` number of words appart.  Another option
would be to put each *sentence* on its own line, in which case two in different
adjacent sentences would not be considered as cooccurring even if they were
fewer than ``--window-size`` words appart.

Explain the basic options, and then point to 

To learn about the other options run ``python run_mle.py -h``.


Train Embeddings:
^^^^^^^^^^^^^^^^^

Hilbert can make embeddings using currently three models: Hilbert-MLE,
Hilbert-SGNS, and Hilert-GloVe.  Usually, you'll make embeddings by invoking
one of the runner scripts found in ``hilbert/hilbert/runners/``.

The minimal commands for running Hilbert-MLE file is:

.. code-block:: bash

    python run_hbt_w2v.py \
        --bigram /path/to/bigram-statistics/
        --out-dir /path/to/vectors \
        --learning-rate 0.025 \
        --epochs 100 \
        --dimensions 300

Where ``/path/to/bigram-statistics`` should point to the directory created by
running the bigram statistics extraction script.  The script will create
embeddings and save them in the directory ``path/to/vectors``.  Some of the 
key parameters for the run are shown.  There are a lot of other options, 
which you can learn about by running ``python run_mle.py -h``.

Running Hilbert-SGNS and Hilbert-GloVe is similar.  Here is a command 
that uses good defaults for Hilbert-SGNS:

.. code-block:: bash

    python run_hbt_w2v.py \
        --bigram /path/to/bigram-statistics/
        --out-dir /path/to/vectors \
        --learning-rate 0.025 \
        --epochs 100 \
        --dimensions 300

Finally, a command that runs Hilbert-GloVe with good defaults:

.. code-block:: bash

    python run_hbt_w2v.py \
        --bigram /path/to/bigram-statistics/
        --out-dir /path/to/vectors \
        --learning-rate 0.025 \
        --epochs 100 \
        --dimensions 300

The runtime depends on the vocabulary size.  Training the top 50k most frequent
words in a concatenation of Gigaword and a Wikipedia 2018 dump takes about
3hrs for each of the models.

.. todo::

    does the output dir need to exist?
    are the defaults generally good numbers?


.. todo::

    What kinds of encodings are supported?  Are there any "bad" inputs that
    should not appear in the corpus?








Use Embeddings
~~~~~~~~~~~~~~

Use a Custom Model
~~~~~~~~~~~~~~~~~~


Reference
~~~~~~~~~




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

In most cases, vectors, as opposed to covectors, are desired.  To access covectors, use the ``get_covec()`` method considered:

.. code-block:: python
    
    >>> embeddings.get_covec(3170)
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])
    >>> embeddings.get_covec('dog')
    tensor([0.4308, 0.9972, 0.0308, 0.6320, 0.6734, 0.9966, 0.7073, 0.2918...])

(or just slice into the ``W`` attribute.)


Reference for ``hilbert.embeddings.Embeddings``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:module:: hilbert

.. autoclass:: embeddings.Embeddings

    .. py:attribute:: V

        A 2D ``torch.Tensor`` with ``dtype=torch.float32`` whose rows
        correspond to word vectors.  If the Embeddings are created with
        ``implementation='numpy'``, then ``V`` will be a ``numpy.ndarray``.

    .. py:attribute:: W

        A 2D ``torch.Tensor`` with ``dtype=torch.float32`` whose rows
        correspond to word covectors. If the Embeddings are created with
        ``implementation='numpy'``, then ``V`` will be a ``numpy.ndarray``.
        Embeddings can be created without any covectors, in which case ``W``
        will be ``None``.

    .. py:attribute:: dictionary

        A ``hilbert.dictionary.Dictionary`` instance, or ``None``, depending on
        whether a dictionary was provided when creating the embeddings.

    .. py:attribute:: normed

        ``True`` if all the vectors in ``V`` and ``W`` (if it exists) are
        normalized, otherwise ``False``.

    .. automethod:: unk()
    .. automethod:: unkV()
    .. automethod:: unkW()
    .. automethod:: save()
    .. automethod:: load()
    .. automethod:: get_vec()
    .. automethod:: get_covec()
    .. automethod:: normalize()
    .. automethod:: check_normalized()
    .. automethod:: greatest_product()
    .. automethod:: greatest_product_one()
    .. automethod:: greatest_cosine()
    .. automethod:: greatest_cosine_one()
    .. automethod:: handle_out_of_vocab()



Reference for ``hilbert.embeddings.random``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: embeddings.random

