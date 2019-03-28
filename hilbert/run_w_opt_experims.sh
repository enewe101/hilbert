
python run_mle.py -b 10000-vocab-lowered/1.4096-8w-dynamic-10k -o diffusion-hilbert/1.4096-10w-mle-l300-I500-V10k --seed 1 -s adam -l 300 -I 500 -t 1 --loader-policy buffered-gpu
python run_mle.py -b 10000-vocab-lowered/1.4096-9w-dynamic-10k -o diffusion-hilbert/1.4096-11w-mle-l300-I500-V10k --seed 1 -s adam -l 300 -I 500 -t 1 --loader-policy buffered-gpu
