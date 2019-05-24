from word_sense import sense_solver
from hilbert.runners import run_base
from hilbert import autotune


def main():
    # parser part
    parser = run_base.get_argparser()
    parser.add_argument(
        '--n-iters', type=int, default=100,
        help='how many cycles to run for each LR tested'
    )
    parser.add_argument(
        '--head-lr', type=float, default=1e5,
        help='estimate of what the maximum possible LR could be'
    )
    '''
        cooccurrence_path,
        shard_factor=1,     # Dense option
        bias=False,
        init_embeddings_path=None,
        dimensions=300,
        sense=5,
        learning_rate=0.01,
        opt_str='adam',
        seed=616,
        device=None,
        verbose=True,
    '''


    filter_kwargs = {'num_writes', 'num_updates', 'n_iters', 'head_lr','batch_size', 'save_embeddings_dir'}


    run_base.add_common_constructor_args(parser)
    run_base.add_shard_factor_arg(parser)
    run_base.add_batch_size_arg(parser)

    parser.add_argument('--sense', type=int, default=5, help='number of senses per word')

    constr_kwargs = parser.parse_args()
    n_iters = constr_kwargs['n_iters']
    head_lr = constr_kwargs['head_lr']

    print(constr_kwargs)

    for kw in filter_kwargs:
        del constr_kwargs[kw]




    constructor = sense_solver.build_sense_mle_solver
    print("Tuning sense embedder!")


    good_lr = autotune.autotune(constructor, constr_kwargs, head_lr=head_lr, n_iters=n_iters)

    print("good learning rate is: ",good_lr)


if __name__ == '__main__':
    main()

