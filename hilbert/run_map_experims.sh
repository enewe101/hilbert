python run_map.py -b 10000-vocab-lowered/1.8-5w-dynamic-10k -o diffusion-hilbert/1.8-5w-map-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered
python run_map.py -b 10000-vocab-lowered/1.32-5w-dynamic-10k -o diffusion-hilbert/1.32-5w-map-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered

python run_map.py -b 10000-vocab-lowered/1.128-5w-dynamic-10k -o diffusion-hilbert/1.128-5w-map-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered

python run_map.py -b 10000-vocab-lowered/1.512-5w-dynamic-10k -o diffusion-hilbert/1.512-5w-map-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered

python run_map.py -b 10000-vocab-lowered/1.2048-5w-dynamic-10k -o diffusion-hilbert/1.2048-5w-map-l250-I200-V10k --seed 1 -s adam -l 250 -I 200 -t 1 --loader-policy buffered

python run_map.py -b 10000-vocab-lowered/1.4096-5w-dynamic-10k -o diffusion-hilbert/1.4096-5w-map-l300-I200-V10k --seed 1 -s adam -l 300 -I 200 -t 1 --loader-policy buffered

python run_map.py -b 10000-vocab-lowered/1.8192-5w-dynamic-10k -o diffusion-hilbert/1.8192-5w-map-l200-I200-V10k --seed 1 -s adam -l 200 -I 200 -t 1 --loader-policy buffered

python run_map.py -b 10000-vocab-lowered/1.16384-5w-dynamic-10k -o diffusion-hilbert/1.16384-5w-map-l200-I200-V10k --seed 1 -s adam -l 200 -I 200 -t 1 --loader-policy buffered

