python run_map.py -b 1.32-5w-dynamic-10k -o diffusion-hilbert/1.32-5w-map-l500-I200-V10k --seed 1 -s adam -l 500 -I 200 -t 1 --loader-policy buffered
python run_map.py -b 1.128-5w-dynamic-10k -o diffusion-hilbert/1.128-5w-map-l500-I300-V10k --seed 1 -s adam -l 500 -I 300 -t 1 --loader-policy buffered
python run_map.py -b 1.512-5w-dynamic-10k -o diffusion-hilbert/1.512-5w-map-l500-I300-V10k --seed 1 -s adam -l 500 -I 300 -t 1 --loader-policy buffered
python run_map.py -b 1.2048-5w-dynamic-10k -o diffusion-hilbert/1.2048-5w-map-l250-I400-V10k --seed 1 -s adam -l 250 -I 400 -t 1 --loader-policy buffered

