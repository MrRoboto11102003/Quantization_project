import numpy as np

layer_names = ['conv1'] + [f'L{g+1}.B{b}' for g, n in enumerate([3,3,3]) for b in range(n)]
print(layer_names)
print(len(layer_names))

mean_bits = np.random.rand(10)
for name, mb in zip(layer_names, mean_bits):
    print(f'  {name}: {mb:.2f}')
