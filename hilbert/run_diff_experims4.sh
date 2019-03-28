python run_diffu.py -b 10000-vocab-lowered/1.8-5w-dynamic-10k -o diffusion-hilbert/1.8-5w-diff2-l600-I200-V10k --seed 1 -s adam -l 600 -I 200 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.32-5w-dynamic-10k -o diffusion-hilbert/1.32-5w-diff2-l600-I200-V10k --seed 1 -s adam -l 600 -I 200 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.128-5w-dynamic-10k -o diffusion-hilbert/1.128-5w-diff2-l600-I300-V10k --seed 1 -s adam -l 600 -I 300 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.512-5w-dynamic-10k -o diffusion-hilbert/1.512-5w-diff2-l500-I400-V10k --seed 1 -s adam -l 500 -I 400 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.2048-5w-dynamic-10k -o diffusion-hilbert/1.2048-5w-diff2-l200-I400-V10k --seed 1 -s adam -l 200 -I 400 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.4096-5w-dynamic-10k -o diffusion-hilbert/1.4096-5w-diff2-l200-I500-V10k --seed 1 -s adam -l 200 -I 500 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.8192-5w-dynamic-10k -o diffusion-hilbert/1.8192-5w-diff2-l150-I600-V10k --seed 1 -s adam -l 150 -I 600 -t 1 -w 2 --loader-policy buffered-gpu
python run_diffu.py -b 10000-vocab-lowered/1.16384-5w-dynamic-10k -o diffusion-hilbert/1.16384-5w-diff2-l80-I600-V10k --seed 1 -s adam -l 80 -I 600 -t 1 -w 2 --loader-policy buffered-gpu
