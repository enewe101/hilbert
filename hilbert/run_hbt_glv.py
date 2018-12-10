import os
import hilbert.run_base as hrun
import hilbert.factories as proletariat


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
        shard_factor=1,
        seed=1,
        device=None,
    ):

    embsolver = proletariat.construct_glv_solver(
        bigram_path=bigram_path, init_embeddings_path=init_embeddings_path,
        d=d, alpha=alpha, xmax=xmax,update_density=update_density,
        learning_rate=learning_rate, opt_str=opt_str,
        shard_factor=shard_factor, seed=seed, device=device
    )

    hrun.init_workspace(embsolver, save_embeddings_dir)
    trace_path = os.path.join(save_embeddings_dir, 'trace.txt')

    # run it up!
    for epoch in range(1, epochs+1):
        print('epoch\t{}'.format(epoch))
        losses = embsolver.cycle(epochs=iters_per_epoch, hold_loss=True)

        # saving data
        hrun.save_embeddings(embsolver, save_embeddings_dir, iters_per_epoch * epoch)
        crt_iter = (epoch - 1) * iters_per_epoch
        hrun.write_trace(trace_path, crt_iter, losses)


if __name__ == '__main__':

    base_parser = hrun.get_base_argparser()
    base_parser.add_argument(
        '--X-max', '-x', type=float, default=100, dest='xmax',
        help="xmax in glove weighting function"
    )
    base_parser.add_argument(
        '--alpha', '-a', type=float, default=3/4,
        help="exponent in the weighting function for glove"
    )
    all_args = vars(base_parser.parse_args())
    hrun.modify_args(all_args)
    run_glv(**all_args)
