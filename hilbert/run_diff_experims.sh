python run_diffu.py -b 10000-vocab-lowered/1.8-1w-dynamic-10k -o diffusion-hilbert/1.8-1w-diff2-l400-I200-V10k --seed 1 -s adam -l 400 -I 200 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.32-1w-dynamic-10k -o diffusion-hilbert/1.32-1w-diff2-l400-I200-V10k --seed 1 -s adam -l 400 -I 200 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.128-1w-dynamic-10k -o diffusion-hilbert/1.128-1w-diff2-l400-I200-V10k --seed 1 -s adam -l 400 -I 200 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.512-1w-dynamic-10k -o diffusion-hilbert/1.512-1w-diff2-l300-I300-V10k --seed 1 -s adam -l 300 -I 300 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.2048-1w-dynamic-10k -o diffusion-hilbert/1.2048-1w-diff2-l100-I300-V10k --seed 1 -s adam -l 100 -I 300 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.4096-1w-dynamic-10k -o diffusion-hilbert/1.4096-1w-diff2-l100-I400-V10k --seed 1 -s adam -l 100 -I 400 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.8192-1w-dynamic-10k -o diffusion-hilbert/1.8192-1w-diff2-l50-I400-V10k --seed 1 -s adam -l 50 -I 400 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.16384-1w-dynamic-10k -o diffusion-hilbert/1.16384-1w-diff2-l50-I400-V10k --seed 1 -s adam -l 50 -I 400 -t 1 -w 2 --loader-policy buffered-gpu

python run_mle.py -b 10000-vocab-lowered/1.2048-1w-dynamic-10k -o diffusion-hilbert/1.2048-1w-mle-l20-I600-V10k --seed 1 -s adam -l 20 -I 600 -t 1 --loader-policy buffered-gpu
python run_mle.py -b 10000-vocab-lowered/1.4096-1w-dynamic-10k -o diffusion-hilbert/1.4096-1w-mle-l20-I600-V10k --seed 1 -s adam -l 20 -I 600 -t 1 --loader-policy buffered-gpu
python run_mle.py -b 10000-vocab-lowered/1.8192-1w-dynamic-10k -o diffusion-hilbert/1.8192-1w-mle-l10-I600-V10k --seed 1 -s adam -l 10 -I 600 -t 1 --loader-policy buffered-gpu
python run_mle.py -b 10000-vocab-lowered/1.16384-1w-dynamic-10k -o diffusion-hilbert/1.16384-1w-mle-l10-I600-V10k --seed 1 -s adam -l 10 -I 600 -t 1 --loader-policy buffered-gpu
