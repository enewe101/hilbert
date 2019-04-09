import hilbert.factories as proletariat
from hilbert.runners.run_base import init_and_run, modify_args, get_base_argparser, kw_filter

def run_w2v(
        cooccurrence_path,
        save_embeddings_dir,
        k=15,
        t_clean_undersample=2.45e-5,
        alpha_smoothing=0.75,
        **kwargs
    ):

    embsolver = proletariat.construct_w2v_solver(
        cooccurrence_path=cooccurrence_path, k=k,
        t_clean_undersample=t_clean_undersample,
        alpha_unigram_smoothing=alpha_smoothing,
        **kw_filter(kwargs)
    )
    init_and_run(embsolver,
                 kwargs['epochs'],
                 kwargs['iters_per_epoch'],
                 kwargs['shard_times'],
                 save_embeddings_dir)

if __name__ == '__main__':

    base_parser = get_base_argparser()
    base_parser.add_argument(
        '--t-clean', '-t', type=float, default=2.45e-5, dest='t_clean_undersample',
        help="Post-sampling (clean) Common word undersampling threshold"
    )
    base_parser.add_argument(
        '--neg-samples', '-k', type=int, default=15, dest='k',
        help="number of negative samples"
    )
    base_parser.add_argument(
        '--alpha-smoothing', '-a', type=float, default=0.75, dest='alpha_smoothing',
        help='context distribution smoothing of PMI'
    )
    all_args = vars(base_parser.parse_args())
    modify_args(all_args)
    run_w2v(**all_args)


"""
Example command:
python run_hbt_w2v.py -u 1.0 -l 0.01 -s adam -I 100 --init std-w2v-s1-t1-v10k-iter5/vectors-init 
--epochs 150 --seed 1 --cooccurrence 5w-dynamic-10k/thresh1 -t 2.45e-05 -o testw2v

"""
