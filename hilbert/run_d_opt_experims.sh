python run_diffu.py -b 10000-vocab-lowered/1.4096-6w-dynamic-10k -o diffusion-hilbert/1.4096-6w-diff1-l250-I500-V10k --seed 1 -s adam -l 250 -I 500 -t 1 -w 1 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.4096-6w-dynamic-10k -o diffusion-hilbert/1.4096-6w-diff2-l300-I600-V10k --seed 1 -s adam -l 300 -I 600 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.4096-6w-dynamic-10k -o diffusion-hilbert/1.4096-6w-diff3-l300-I600-V10k --seed 1 -s adam -l 300 -I 600 -t 1 -w 3 --loader-policy buffered-gpu

python run_diffu.py -b 10000-vocab-lowered/1.8-6w-dynamic-10k -o diffusion-hilbert/1.8-6w-diff1-l650-I300-V10k --seed 1 -s adam -l 650 -I 300 -t 1 -w 1 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.8-6w-dynamic-10k -o diffusion-hilbert/1.8-6w-diff2-l700-I300-V10k --seed 1 -s adam -l 700 -I 300 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.8-6w-dynamic-10k -o diffusion-hilbert/1.8-6w-diff3-l750-I300-V10k --seed 1 -s adam -l 750 -I 300 -t 1 -w 3 --loader-policy buffered-gpu
