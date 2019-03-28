python run_mle.py -b 10000-vocab-lowered/1.8192-5w-dynamic-10k -o diffusion-hilbert/1.8192-5w-mle-l100-I500-V10k --seed 1 -s adam -l 100 -I 500 -t 1 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.8192-5w-dynamic-10k -o diffusion-hilbert/1.8192-5w-diff1-l100-I500-V10k --seed 1 -s adam -l 100 -I 500 -t 1 -w 1 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.2048-1w-dynamic-10k -o diffusion-hilbert/1.2048-1w-diff1-l50-I400-V10k --seed 1 -s adam -l 50 -I 400 -t 1 -w 1 --loader-policy buffered-gpu
