import os
import argparse
import hilbert as h


def absolutize_paths(args):
    if h.CONSTANTS.RC['corpus_dir'] is not None:
        args['corpus_path'] = os.path.join(
            h.CONSTANTS.RC['corpus_dir'], args['corpus_path']
        )

    if h.CONSTANTS.RC['cooccurrence_dir'] is not None:
        args['save_path'] = os.path.join(
            h.CONSTANTS.RC['cooccurrence_dir'], args['save_path']
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "Extracts unigram and cooccurrence statistics from a corpus, "
        "and stores to disk"
    ))
    parser.add_argument(
        '--corpus', '-c', required=True, dest='corpus_path',
        help=(
            "File name for input corpus.  If you have specified a "
            "``corpus_dir`` in your ~/.hilbertrc, then relative paths will be "
            "interpreted as relative to your corpus_dir.  Use an absolute "
            "path to override."
        )
    )
    parser.add_argument(
        '--out-dir', '-o', dest='save_path', required=True, 
        help=(
            "Name of directory in which to store cooccurrence data.  It will "
            "be created if it does not exist.  If you "
            "have specified a ``corpus_dir`` in your ~/.hilbertrc file "
            "then relative paths will be interpreted relative to your "
            "``corpus_dir``. Use an absolute path to override."
        )
    )
    parser.add_argument(
        '--extractor', '-e', dest='extractor_str',
        help="Type of sampler to use",
        choices=('flat', 'harmonic', 'dynamic', 'custom'),
        required=True
    )
    parser.add_argument(
        '--window', '-w', help="Cooccurrence window size",
        default=None, type=int
    )
    parser.add_argument(
        '--weights-file', '-f', type=str, default=None, help=(
            "File containing cooccurrence weighting scheme.  see {} for an "
            "example. Put right weights on the first line, as floats "
            "separated by spaces.  Put left-weights on the second line. "
            "Weights should be ordered so that increasing index represents "
            "greater distance from the focal word.  I.e., left-weights should "
            "seem reversed compared to the order in which "
            "they are applied in the corpus.".format(
                h.cooccurrence.extractor.WEIGHTS_EXAMPLE_PATH)
        )
    )
    parser.add_argument(
        '--vocab', '-v', type=int, default=None,
        help="Prune vocabulary to the most common VOCAB number of words"
    )
    parser.add_argument(
        '--processes', '-p', help="Number of processes to spawn",
        default=1, type=int
    )
    parser.add_argument(
        '--min-count', '-m', default=None, type=int,
        help="Minimum number of occurrences below which token is ignored",
    )
    parser.add_argument(
        '--quiet', '-q', dest='verbose', default=True, action='store_false',
        help="Don't print to stdout during execution."
    )

    # Parse the arguments
    args = vars(parser.parse_args())
    absolutize_paths(args)
    weights_file = args.pop('weights_file')
    if weights_file is not None:
        args['weights'] = h.cooccurrence.extractor.read_weights_file(
                weights_file)

    h.cooccurrence.extraction.extract_unigram_and_cooccurrence(**args)

