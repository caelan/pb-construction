# pb-construction
Pybullet Construction Planning

## Installation

```
$ git clone https://github.com/caelan/pb-construction.git
$ cd pb-construction
$ git submodule update --init --recursive
$ ./pddlstream/FastDownward/build.py
```

To build ikfast modules:
```bash
$ cd utils/ikfast/kuka_kr6_r900/
$ python setup.py build
```

For other robots, replace `kuka_kr6_r900` with the following supported robots:
- `eth_rfl`
- `abb_irb6600_track`

## Examples

* `$ python -m extrusion.run`
* `$ python -m picknplace.run`

## Testing new IKFast modules


* `$ python -m debug_examples.test_eth_rfl_pick.py`
* `$ python -m debug_examples.test_irb6600_track_pick.py`

