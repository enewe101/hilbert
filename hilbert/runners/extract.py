import hilbert as h


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "Extracts unigram and bigram statistics from a corpus, "
        "and stores to disk"
    ))
    parser.add_argument(
        '--corpus', '-c', required=True, dest='corpus_filename',
        help="File name for input corpus"
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, 
        help="Name of directory in which to store cooccurrence data"
    )
    parser.add_argument(
        '--thresh', '-t', type=float, help=(
            "Threshold for common-word undersampling, "
            "for use with w2v sampler only"
        )
    )
    parser.add_argument(
        '--sampler', '-s', help="Type of sampler to use",
        choices=('w2v', 'flat', 'harmonic', 'dynamic'), required=True,
        dest="sampler_type"
    )
    parser.add_argument(
        '--window', '-w', help="Cooccurrence window size",
        required=True, type=int
    )
    parser.add_argument(
        '--vocab', '-v', type=int, default=None,
        help="Prune vocabulary to the most common `vocab` number of words"
    )
    parser.add_argument(
        '--processes', '-p', help="Number of processes to spawn",
        default=1, type=int
    )
    parser.add_argument(
        '--min-count', '-m', default=None, type=int,
        help="Minimum number of occurrences below which token is ignored",
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Corpus path and output directory are relative to standard locations.
    args['corpus_path'] = os.path.join(
        shared.CONSTANTS.TOKENIZED_CAT_DIR, args['corpus_filename']
    )
    del args['corpus_filename']
    args['out_dir'] = os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['out_dir']
    )

    if args['min_count'] is not None and args['vocab'] is not None:
        raise ValueError('Use either --vocab or --min-count, not both.')

    # thresh should only be specified if the sampler is w2v
    if args['thresh'] is not None and args['sampler_type'] != 'w2v':
        raise ValueError('thresh argument is only valid for w2v sampler.')

    h.bigram.bigram_extraction.extract_unigram_and_bigram(**args)

