python run_mle.py -b 2.8-5w-dynamic-10k -o diffusion-hilbert/1.8-5w-mle-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered
python run_mle.py -b 1.32-5w-dynamic-10k -o diffusion-hilbert/1.32-5w-mle-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered
python run_mle.py -b 1.128-5w-dynamic-10k -o diffusion-hilbert/1.128-5w-mle-l500-I300-V10k --seed 1 -s adam -l 500 -I 300 -t 1 --loader-policy buffered
python run_mle.py -b 1.512-5w-dynamic-10k -o diffusion-hilbert/1.512-5w-mle-l500-I300-V10k --seed 1 -s adam -l 500 -I 300 -t 1 --loader-policy buffered
