import os
import hilbert.run_base as hrun
import hilbert.factories as proletariat


def run_w2v(
        bigram_path,
        save_embeddings_dir,
        epochs=100,
        iters_per_epoch=100,
        init_embeddings_path=None,
        d=300,
        k=15,
        t_clean_undersample=None,
        alpha_smoothing=0.75,
        update_density=1.,
        learning_rate=0.01,
        opt_str='adam',
        sector_factor=1,
        shard_factor=1,
        shard_times=1,
        seed=1,
        device=None,
    ):

    embsolver = proletariat.construct_w2v_solver(
        bigram_path=bigram_path, init_embeddings_path=init_embeddings_path,
        d=d, k=k, t_clean_undersample=t_clean_undersample,
        alpha_unigram_smoothing=alpha_smoothing, update_density=update_density,
        mask_diagonal=mask_diagonal, learning_rate=learning_rate,
        opt_str=opt_str, shard_factor=shard_factor,
        sector_factor=sector_factor, num_loaders=num_loaders,
        queue_size=queue_size, loader_policy=loader_policy,
        seed=seed, device=device
    )

    print(embsolver.describe())
    hrun.init_workspace(embsolver, save_embeddings_dir)
    trace_path = os.path.join(save_embeddings_dir, 'trace.txt')

    # run it up!
    for epoch in range(1, epochs+1):
        print('epoch\t{}'.format(epoch))
        losses = embsolver.cycle(
            iters=iters_per_epoch, shard_times=shard_times)

        # saving data
        hrun.save_embeddings(
            embsolver, save_embeddings_dir, iters_per_epoch * epoch)
        crt_iter = (epoch - 1) * iters_per_epoch
        hrun.write_trace(trace_path, crt_iter, losses)



if __name__ == '__main__':

    base_parser = hrun.get_base_argparser()
    base_parser.add_argument(
        '--t-clean', '-t', type=float, default=None, dest='t_clean_undersample',
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
    hrun.modify_args(all_args)
    run_w2v(**all_args)


"""
Example command:
python run_hbt_w2v.py -u 1.0 -l 0.01 -s adam -I 100 --init std-w2v-s1-t1-v10k-iter5/vectors-init 
--epochs 150 --seed 1 --bigram 5w-dynamic-10k/thresh1 -t 2.45e-05 -o testw2v

"""
