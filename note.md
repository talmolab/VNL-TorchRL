# Software Stack Overviews

| Low Level -> High Level | `MuJoCo` | `MuJoCo-MJX` | `dm_control/mjcf` | `dm_control` | `brax` | `torch-rl` | `brax/training` | `acme` + `launchpad` |
|---|---|---|---|---|---|---|---|---|
| **Purpose** | Physic Simulation | GPU Accelerated PhySim | xml manipulation | env/task | env/task | RL Training | RL Training | RL Training + Distributed |
| **Pros** | - | - | Easy access to xml file element, named element more sophisticated reward and observation  engineering | Have predefined environments in the deep neuroethology paper.  Can Directly use `mjcf`. | Native support for `mujoco-mjx` <br> Seems to be more sample efficient in certain algorithm implementation such as `ppo` | Easy to use, good documentation and community supports | Everything in JAX, significant speed up. | what they used in google in the deep neuroehtology paper. |
| **Cons** | We have no choice | We have no choice | Heavily depends on `dm_control`. <br> It works when the system is fully on dm_control  | `acme` did not properly execute / hard to modify. <br> did not natively support  `mujoco-mjx`. | Still in early development. The nature of compilation +  model complexity makes it harder to debug. We need to re-invent the wheel  of `dm_control/mjcf` | Overhead in converting JAX->torch tensor in every env steps. <br> Need to reinvent the wheel, to adapt `dm_control/mjcf` and `mujoco-mjx` | Did not have a reliable checkpoint system yet, which means that we cannot have reproducible results. | No Documentation, elusive wrappers,  open source but not open hardware protocols |


## Approachable Paths

|  | `MuJoCo-MJX` | `Brax` | `Torch-RL` | **Proposed Solutions** |
|---|---|---|---|---|
| **Purpose** | Acc. PhySim  | Env/Task <br>Wrap mjx layer | RL Training | 1. Ditch `Brax`, directly warp the env using `torch` structure<br>`MuJoCo + Torch-RL` or `MuJoCo-MJX + torchrl`<br><br>_Implications:_ Need time/effort to adapt APIs, but will be relatively robust to changes because we did not depend on `brax` (from google) by `torchrl` (by meta) anymore |
| **Advantage** | Efficient & Easy to Scale | OOB support for MJX | Transparent/Flexible Arch.<br>Easier NN Modifications<br>Good Community Supports | 2. Ditch `torchrl`, use `mujoco-mjx + brax + brax/training`<br><br>_Implications:_ Mostly under same JAX arch and API should be consistent, but need to learn functional programming and JAX to enable custom NN arch.  |
| **Road Block** | - | Huge Overhead in tensor conversion | API Change in Brax+Torch Inconsistent |  |

---
---
---

## Regarding Using TorchRL with Brax

Each step's statistics are forced to convert from JAX tensor to torch tensor on GPU. The overhead might be potentially very high.

The torch-rl provided brax wrapper but it does not manipulate the states and reward and observation nicely into the torchrl module.

There is a million things that makes the products of two tech giant companies, google and meta, to work together. We are stuck with MuJoCo.


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

This error message is caused by `mmap` (memory map) an empty file, where the `self.size` might be 0. Might have to do with some of the training script and how brax's environment stepping API (how it stores the state, pos, reward, etc.) did not interact well with the replay-buffer in `torchrl` 

#### Re-accessing the importance of Brax

Brax sits between the simulator (low level physic sim) and the learning layer (high level RL). In this case of using `mujoco-mjx`, I don't think `brax` provide us with any goods other than injecting random unnecessary API between `brax` and `torchrl`. We might be able to directly write wrappers for the `mujoco-mjx` and let the `torchrl` directly interact with the `mujoco-mjx`.

### Designing Good Reward Function

Use the [mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc)'s predictive sampling (like MCTS in real time) to engineer a good reward function for our rodent fellas. 


> TODO: We need to inject our rodent model into the module and mess around with the nobs like in the demo.

