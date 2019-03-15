# pb-construction
Pybullet Construction Planning

## Installation

```bash
$ git clone https://github.com/caelan/pb-construction.git
$ cd pb-construction
$ git submodule update --init --recursive
$ ./pddlstream/FastDownward/build.py
```

To build ikfast modules:
```bash
$ cd conrob_pybullet/utils/ikfast/kuka_kr6_r900/
$ python setup.py build
```

For other robots, replace `kuka_kr6_r900` with the following supported robots:
- `eth_rfl`
- `abb_irb6600_track`

Construction Sequencing Structural Analysis (for frame structures)
* https://github.com/yijiangh/conmech

## Examples

* `$ python -m extrusion.run`
* `$ python -m picknplace.run`

## Testing new IKFast modules

* `$ python -m conrob_pybullet.debug_examples.test_eth_rfl_pick`
* `$ python -m conrob_pybullet.debug_examples.test_irb6600_track_pick`

## Related Repos

* https://github.com/yijiangh/pychoreo
* https://github.com/yijiangh/assembly_instances
* https://github.com/yijiangh/conmech
* https://github.com/yijiangh/conrob_pybullet
