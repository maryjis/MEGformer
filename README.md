



## Requirements

You can create a new conda environment and install the required dependencies
```shell
conda create -n bm ipython python=3.8 -y
conda activate bm
conda install pytorch torchaudio cudatoolkit=11.3 -c pytorch -y
pip install -U -r requirements.txt
pip install -e .


## Data & Studies


- `audio_mous` (Scheffelen2019 in the paper): MEG, 273 sensors, 96 subjects, 80.9 hours, [reference](https://www.nature.com/articles/s41597-019-0020-y). 
- `gwilliams2022` : MEG, 208 sensors, 27 subjects, 56.2 hours, [reference](https://www.nature.com/articles/s41467-022-34326-1).



### Load dataset


dora run download_only=true 'dset.selections=[gwilliams2022]'


## Training

To start training, from the root of the repository, run
```bash
dora run [-f SIG] [-d] [--clear] [ARGS]
```

The `-f SIG` flag will inject all the args from the XP with the given signature first, and then
complete with the one given on the command line.

`--clear` will remove any existing XP folder, checkpoints etc.

`[ARGS]` is a list of overrides for the Hydra config. See [conf/config.yaml](conf/config.yaml) for a list of all parameters.

```

### Grid files

Larger scale experiments should be conducted within grid files. Grid files are defined in `bm/grids/` as normal python files. Define all the XP you want:

If the file is called `bm/grids/mygrid.py`, run
```bash
dora grid mygrid
```
This will schedule all required XPs. If you change the experiments defined in the
file, it will cancel any job that is no longer required, and schedule any new
job required (e.g., you can cancel a job by commenting the corresponding line).

Some useful flags:
- `-r`: retry all failed or cancelled XPs.
- `-C`: cancel all jobs.
- `-T{IDX}`: trim all metrics to match the number of epochs of the XP
	with the given index in the grid.
- `-t{IDX}`: show the log from the XP with the given index in the grid.

If you want to restart from scratch (you changed some important code), then either
use the `dummy` parameter in the config in the top level `bind_()`, or use the `--clear` flag:
```
dora grid mygrad --clear
```
This will ask confirmation first, because this is quite a dangerous command!!

### Grids for reproducing our paper

The main results can be reproduced with `dora grid nmi.neuro_experiments_cnntransformer.py`.
Checkout the [grids folder](./bm/grids/) for the available grids,



```

## Evaluations

The evaluation requires running a separate script. Once a grid is fully trained,
you can run the evaluation on it with

```bash
python -m scripts.run_eval_probs grid_name="main_table"
```
