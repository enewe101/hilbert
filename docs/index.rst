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
training.  In ``hilbert/runners/`` do:

.. code-block:: bash
    
    python extract.py \
        --corpus /path/to/corpus.txt \
        --out-dir /path/to/cooccurrence-statistics/ \
        --sampler dynamic \ 
        --window 5 \
        --vocab 50000 \
        --processes 8 \


Your input corpus should be space-tokenized and have one document per line.
Run ``python extract.py -h`` for an explanation of all options.  Read on for
more details about how to format your input corpus, and how cooccurrence
statistics.

The second script uses the bigram statistics to train embeddings
according to your choice of model, and writes the embeddings to disk.  To run
the Hilbert-MLE model, go to ``hilbert/runners/`` and do:

.. code-block:: bash
    
    python run_mle.py \
        --bigram /path/to/cooccurrence-statistics/ \
        --out-dir /path/to/vectors \
        --learning-rate 0.025 \
        --epochs 100 \
        --dimensions 300

Note that once you've extracted bigram data once, you can use it to train many
different models.  You can run Hilbert-SGNS and Hilbert-GloVe by substituting
``run_hbt_w2v.py``, or ``run_hbt_glv.py`` respectively in place of
``run_mle.py`` above.

Read on for more details about these commands.


Extracting Cooccurrence Statistics:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``runners/extract.py`` script reads a corpus file, and extracts the
vocabulary, the unigram statistics, and the cooccurrence statistics---the
number of times word $i$ occurs near word $j$, for each $(i,j)$ pair, for some
definition of "cooccurrence".  There are different ways to define cooccurrence,
but they all start by setting some ``--window`` size, such that words separated
by fewer than ``--window`` tokens are considered to cooccur according to some weighting scheme that depends on the ``--sampler`` chosen.

This script will write files to the the location specified by ``--out-dir``
(which will be made if it doesn't exist).  A vocabulary will be stored at
``out_dir/dictionary``, unigram frequencies will be stored in
``out_dir/Nx.txt``, and cooccurrence statistics will be stored in various
``out_dir/*.npz`` files.  These are basically serializations instances of the
``dictionary.Dictionary``, ``unigram.Unigram`` and
``bigram.bigram_base.BigramBase`` classes.  See the reference for those
classes to understand the formats and how they work.


Sampler
.......

The ``runners/extract.py`` script reads a corpus file, and extracts 
cooccurrence statistics---the number of times word $i$ occurs near word $j$,
for each $(i,j)$ pair, for some definition of "cooccurrence".  There are 
different ways to define cooccurrence, but they all start by setting some
``--window`` size, such that words separated by fewer than ``--window`` tokens
are considered to cooccur.  Where they differ is in how cooccurrences are
weighted.

Using ``--sampler flat`` always attributes a cooccurrence weight of 1.  Using
``--sampler dynamic``, counts cooccurrences in a way that is similar to SGNS,
except that it deterministically takes the weighting equal to the expectation
of SGNS's stochastic sampler; thus, it assigns a cooccurrence weight of
``(separation+1)/window_size``, where ``separation`` is the distance, in
tokens, between cooccurring tokens.  Using ``--sampler harmonic`` counts
cooccurrences identically to GloVe, assigning a cooccurrence weight of
``1/separation``.


Corpus file format
..................
The corpus file should be in the following format: tokens should be
space-separated, and documents should be line-separated.  I.e., place each
document on its own line, in one large file.  Normally, you should lower-case
the corpus, unless you want to obtain, for example, unique embeddings for
"Apple" and "apple".

The script considers cooccurrence separately within each line, i.e. words near
the end of one line are never considered to cooccur with words near the
beginning of the next line.  Normally it makes sense to put each document on a
line.  Depending on your needs, you may want to do something else, like put
each sentence on its own line, in which case only tokens within the same
sentence can be considered to cooccur.


Train Embeddings:
^^^^^^^^^^^^^^^^^

Hilbert can currently make embeddings using one of three models: Hilbert-MLE,
Hilbert-SGNS, and Hilert-GloVe.  Usually, you'll make embeddings by invoking
one of the runner scripts found in ``hilbert/hilbert/runners/``.

The command for running Hilbert-MLE using good hyperparameter defaults was
quoted above.  And we recommend using that because of its stable training
dynamics and good all-round performance.  But, if you would like to run
Hilbert-SGNS with good defaults, by all means do:

.. code-block:: bash

    python run_hbt_w2v.py \
        --bigram /path/to/bigram-statistics/
        --out-dir /path/to/vectors/ \
        --learning-rate 0.025 \
        --epochs 100 \
        --batches 100 \
        --dimensions 300

...and to run Hilbert-GloVe with good defaults do:

.. code-block:: bash

    python run_hbt_w2v.py \
        --bigram /path/to/bigram-statistics/
        --out-dir /path/to/vectors/ \
        --learning-rate 0.025 \
        --epochs 100 \
        --batches 100 \
        --dimensions 300

Where ``/path/to/bigram-statistics`` should point to the directory created by
running the bigram statistics extraction script.  The script will create
embeddings and save them in the directory ``path/to/vectors/``.  Again, to
learn about all of the options, run ``python run_mle.py -h``.

The runtime depends on the vocabulary size.  Training the top 50k most frequent
words in a concatenation of Gigaword and a Wikipedia 2018 dump takes about
3hrs for each of the models.


Dense vs. Sample-based
......................
There are two alternative ways to organize the training of a word embedding
using ``hilbert``: a *dense* approach and a *sample-based* approach.  
These approaches differ in their notion of a batch and an epoch.

To begin it helps to recall that we can take the viewpoint that linear word
embedding is matrix factorization.  Suppose that we take the inner product
between a matrix $W$, consisting of all of the covectors as rows, and a vector
$V$ consisting of all of the vectors as columns.  Then, $(WV)_{i,j} = \langle i
| j \rangle$, and the loss can be written as $\mathcal{L} = \sum f((VW)_{i,j},
M_{i,j})$, where $M$ is some matrix of target association scores which need
to be learned by the embeddings.

In general, the matrix $M_{ij}$ won't fit into memory all at once.  But, 
because the loss function is being taken as a sum across elements of the 
target matrix $M$ and the embedding product matrix $VW$, we are always
free to spread the calculation of terms in this sum across multiple batches.

The most obvious way to do this, the *dense* approach, loads *shards* of these
two matrices into memory.  We do this by selecting a subset of rows
and a subset of columns, and then loading all cells that lie at the intersection
of these rows and columns.  Notice that our *shard* will always be a rectange
of shape ``num_rows * num_columns``.

In the dense approach, we break up the large matrices into shards that 
fully cover the original matrices, and once we have computed the loss on all
individual patches, then we have computed the loss for the entire matrix.

Alternatively, rather than selecting row and column subsets, which always
gives us rectangular shaped patches, we can just sample individual cells in an
unrestricted way---unrestricted in the sense that their intersections aren't
the cartesian product of subsets of rows and columns.  In general, its possible
that only one cell from a given row or column appears in a given sample.

The sample-based approach seems less efficient in the sense that it doesn't
make use of the compact matrix data-structure.  We need to specify every
$(i,j)$ pair rather than just the individual $I$, $J$ subsets.  

However, the sample-based approach is more flexible.  It allows us to leverage
the fact cells in the target matrix are not equally informative when estimating
the loss and its gradient.  For example, for very rare words $i$ and $j$, we
have little statistical evidence on which to base the target association value
(e.g. PMI), and so it might make sense to consider them less often 
when estimating the gradient.  


Epochs and batches
..................
In the dense approach, each shard is essentially a minibatch, except that
rather than batching by subsampling a list of examples, we're subsampling
in two dimensions---rows and columns.  We can choose shards to be as
large as our GPU memory will allow.  Once we have computed the loss across
all shards, it makes sense to say that we have completed one "epoch", because
we have used all of the information in the full dataset to make updates.
but unlike in other modelling contexts one epoch can be executed very quickly,
because of how compactly the matrix represents all of the cooccurrence
statistics from a large corpus.  For this reason, we consider one epoch in the
dense approach to constitute a single meaningful update (although technically
an update is executed on each shard indivicually).

In the sample-based approach, there is really no notion of "an epoch", 
because we don't systematically sample all cells in the matrix.  Even after 
updates across many many samples, there may be cells in the matrix that are
never accessed---most likely those deemed not very important.  So, in this
approach we consider a single sample to constitute a meaningful update,
and we eschew any notion of "epoch" altogether.


Writes, Updates-Per-Write, and Batch-Size
`````````````````````````````````````````
In the end, all that matters is that you can control how long your model 
trains, how often you save intermediate results to disk, and how much data you
can stick on the GPU at once.

Therefore we parametrize the batching behavior in terms of concepts that 
isolate these needs.  We have (1) the total number of meaningful updates to do
(2) the frequency with which we write to disk, and (3) the size of data to 
load onto the GPU at once.


===========================  ===================  =============
Concept                      Dense Option         Sample Option
===========================  ===================  =============
(1) number of writes         --writes             --writes
(2) total number of updates  --updates            --updates
(3) GPU size factor          --shard-factor       --batch-size
===========================  ===================  =============

Notice that the number of writes has nothing to do with long you run the
model.  One write will happen every ``num_updates / num_writes``.  Normally
just leave this at 100 to be able to see how your model performance changes
throughout training.

In the dense case, ``--updates`` corresponds to what we could
reasonably call the number of epochs.  But for the sample-based approach, 
this is the number of individual samples we draw.

Finally, for saturating GPU memory, we rely on a ``--shard-factor`` in the 
dense case, and a ``--batch-size`` in the sample-based case.  The shard factor
the number of row-subsets or number of column-subsets into which we divide
the matrix.  Notice that this means that there will be $shard_factor^2$ number
of shards.  Make this number *bigger* if you're running out of GPU memory.
In the sample based case, the ``--batch-size`` is literally
the number of matrix cells we sample each time.  Make this *smaller* if you 
run out of GPU memory.






.. todo::
    default learning rate and number of epochs for each model

.. todo::
    When you run the extraction, you still need to sectorize.  Or do you.

.. todo::

    What kinds of encodings are supported?  Are there any "bad" inputs that
    should not appear in the corpus?


Evaluate Embeddings
~~~~~~~~~~~~~~~~~~~
Once you've got your embeddings trained, you'll probably want to check whether
they're any good.  We include a script that you can run to check the embeddings
against several *intrinsic* word similarity and anology tasks.  Be warned that
performance on intrinsic tasks does not necessarily correlate with whatever
application you may have in mind, but it is a good check that the embeddings
have learned.  Think of it as a necessary but not sufficient check.  You'll
still need to check against your specific application, but this will usually
catch any fundamental issue with the training.


Use Embeddings
~~~~~~~~~~~~~~


Use a Custom Model
~~~~~~~~~~~~~~~~~~
``hilbert`` makes it easy to define models that are part of the *simple
embedder* class---roughly this means that they amount to matrix factorization.
To make your own embedding model, you'll need to write at least two things:

    1. A data loader.  The type and shape of data to be loaded onto the GPU
           depends on the model, so you'll need to write a loader that puts
           the right data onto the GPU for your model.
    2. A loss function.  No surprise there.

Once you've written these classes, you can optionally write a runner script
which just makes it more convenient to run the model from the command line, by
following the example of the runners for the other models in
``hilbert/runners/``.

In general, there is a potentially big difference between how a model is stated
mathematically, and how a model is implemented.  The fact that the implicit
factorization implementation of SGNS in word2vec seems so much different from
the explicit factorization implementation here is testament to this fact.

What this means for you is that, when going from model to implementation,
you'll need to think about things like memory and data transfer, and you'll
want to take advantage of sparsity in the structure of the problem.


Reference
~~~~~~~~~
``bigram.BigramBase``
``unigram.Unigram``
``dictionary.Dictionary``
``embeddings.Embeddings``




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

