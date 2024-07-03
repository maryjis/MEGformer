



## Requirements

You can create a new conda environment and install the required dependencies
```shell
conda create -n bm ipython python=3.8 -y
conda activate bm
conda install pytorch torchaudio cudatoolkit=11.3 -c pytorch -y
pip install -U -r requirements.txt
pip install -e .
```

## Data & Studies


- `audio_mous` (Scheffelen2019 in the paper): MEG, 273 sensors, 96 subjects, 80.9 hours, [reference](https://www.nature.com/articles/s41597-019-0020-y). 
- `gwilliams2022` : MEG, 208 sensors, 27 subjects, 56.2 hours, [reference](https://www.nature.com/articles/s41467-022-34326-1).



### Load dataset
```

dora run download_only=true 'dset.selections=[gwilliams2022]'

```

### Grid files

Larger scale experiments should be conducted within grid files. Grid files are defined in `bm/grids/` as normal python files. Define all the XP you want:

If the file is called `bm/grids/mygrid.py`, run
```bash
dora grid mygrid
```

### Grids examples of using

The main results can be reproduced with `dora grid nmi.neuro_experiments_cnntransformer  --dry_run --init `.
Checkout the [grids folder](./bm/grids/) for the available grids,


### Training

To start training, from the root of the repository, run
```bash
dora run [-f SIG] [-d] [--clear] [ARGS]
```

The `-f SIG` flag will inject all the args from the XP with the given signature first, and then
complete with the one given on the command line.

`--clear` will remove any existing XP folder, checkpoints etc.

`[ARGS]` is a list of overrides for the Hydra config. See [conf/config.yaml](conf/config.yaml) for a list of all parameters.

### Evaluations

The evaluation requires running a separate script. Once a grid is fully trained,
you can run the evaluation on it with

```bash
python -m scripts.run_eval_probs grid_name="neuro_experiments_cnntransformer"
```

## Reproduce MEGformer results on Gwilliams dataset on 3s segment
```
dora grid nmi.neuro_experiments_cnntransformer  --dry_run --init
dora run -f c97c100b
python -m scripts.run_eval_probs sigs=[c97c100b]
```
