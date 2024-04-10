## Regarding Using TorchRL with Brax

Each step's statistics are forced to convert from JAX tensor to torch tensor on GPU. The overhead might be potentially very high.

The torch-rl provided brax wrapper but it does not manipulate the states and reward and observation nicely into the torchrl module.

### Debugging Progress: 
In running `torch_run.py`, when extending the replay buffer of the data, I encounter the following issue:

```
data_buffer.extend(data_reshape)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/torchrl/data/replay_buffers/replay_buffers.py", line 1040, in extend
    index = super()._extend(tensordicts)
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/torchrl/data/replay_buffers/replay_buffers.py", line 495, in _extend
    index = self._writer.extend(data)
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/torchrl/data/replay_buffers/writers.py", line 267, in extend
    self._storage[index] = data
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/torchrl/data/replay_buffers/storages.py", line 112, in __setitem__
    ret = self.set(index, value)
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/torchrl/data/replay_buffers/storages.py", line 720, in set
    self._init(data[0])
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/torchrl/data/replay_buffers/storages.py", line 1115, in _init
    out = out.memmap_like(prefix=self.scratch_dir)
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/base.py", line 2221, in memmap_like
    return input._memmap_(
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/_td.py", line 1882, in _memmap_
    dest._tensordict[key] = value._memmap_(
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/_td.py", line 1882, in _memmap_
    dest._tensordict[key] = value._memmap_(
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/_td.py", line 1907, in _memmap_
    _populate()
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/_td.py", line 1898, in _populate
    dest._tensordict[key] = MemoryMappedTensor.from_tensor(
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/memmap.py", line 177, in from_tensor
    handler = _FileHandler(size)
  File "/home/scotty/anaconda3/envs/torch-rl/lib/python3.10/site-packages/tensordict/memmap.py", line 663, in __init__
    self.buffer = mmap.mmap(self.fd, self.size)
ValueError: cannot mmap an empty file
```