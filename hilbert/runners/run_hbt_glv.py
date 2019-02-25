import hilbert.factories as proletariat
from hilbert.runners.run_base import init_and_run, modify_args, get_base_argparser

def run_glv(
        bigram_path,
        save_embeddings_dir,
        epochs=100,
        iters_per_epoch=100,
        init_embeddings_path=None,
        d=300,
        xmax=100,
        alpha=0.75,
        update_density=1.,
        learning_rate=0.01,
        opt_str='adam',
        sector_factor=1,
        shard_factor=1,
        shard_times=1,
        seed=1,
        device=None,
        nobias=False,
    ):

    embsolver = proletariat.construct_glv_solver(
        bigram_path=bigram_path, init_embeddings_path=init_embeddings_path,
        d=d, alpha=alpha, xmax=xmax,update_density=update_density,
        learning_rate=learning_rate, opt_str=opt_str, shard_factor=shard_factor,
        sector_factor=sector_factor, seed=seed, device=device, nobias=nobias,
    )
    init_and_run(embsolver, epochs, iters_per_epoch, shard_times, save_embeddings_dir)


if __name__ == '__main__':

    base_parser = get_base_argparser()
    base_parser.add_argument(
        '--X-max', '-x', type=float, default=100, dest='xmax',
        help="xmax in glove weighting function"
    )
    base_parser.add_argument(
        '--alpha', '-a', type=float, default=3/4,
        help="exponent in the weighting function for glove"
    )
    base_parser.add_argument(
        '--nobias', action='store_true',
        help='set this flag to override GloVe defaults and remove bias learning' 
    )
    all_args = vars(base_parser.parse_args())
    modify_args(all_args)
    run_glv(**all_args)
