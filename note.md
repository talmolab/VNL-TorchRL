## Regarding Using TorchRL with Brax

MJX simulation is fast, but each step's statistics are forced to convert from JAX tensor to torch tensor on GPU. The overhead might be potentially very high.