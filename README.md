# RLZilatran

## Slurm Usage

To get the project running using the Slurm Workload manager and conda first ensure that poetry does not currently have a venv (In case remove it.). Then create a conda environment (default is *rlzilatran*) using ac compatible python version. Then simply let poetry install to said env.

This necessitates a conda environment with the following requirements:
```
pip
gcc_linux-64
gxx_linux-64
gfortran_linux-64
poetry>=2.1.0
mpi4py>=4.0.3,<5.0.0
```

mpi4py uses an existing installation since its compilation/configuration is usually specific to the system.

Then the other packages can be installed

```bash
poetry config virtualenvs.create false --local
conda activate rlzilatran # or your conda environment
poetry install
```
Usage is then centralized to [Submission Script](submit.py) relying on the [slurm.toml](slurm.toml) for base config.

## Running Jobs with `submit.py`

[submit.py](submit.py) automates generating and submitting SLURM batch scripts.

### Requirements

* [slurm.toml](slurm.toml) config file defining cluster and job parameters
* Jinja2 template [sbatch_template.j2](sbatch_template.j2) for SLURM script rendering
* Conda environment(s) configured as needed
* **Python Dependencies**:

  * [`click`](https://click.palletsprojects.com/en/stable/): For building command-line interfaces.
  * [`toml`](https://pypi.org/project/toml/): For parsing TOML configuration files.
  * [`jinja2`](https://jinja.palletsprojects.com/): For templating SLURM scripts.

### Usage

```bash
python submit.py [OPTIONS] JOBNAME
```

* **JOBNAME**: Unique name for the job and output directory. All other options are best handled by the [slurm.toml](slurm.toml) unless prototyping where cli options can be used.

### Options Overview

| Option                          | Description                                    |
| ------------------------------- | ---------------------------------------------- |
| `-t, --time`                    | Job time limit (hours)                         |
| `-n, --nodes`                   | Number of compute nodes                        |
| `-q, --queue`                   | SLURM partition/queue                          |
| `--conda_env`                   | Conda environment to activate                  |
| `-apo, --agents-per-observer`   | Number of agents per observer (worker process) |
| `-ntc, --threads-cpu-process`   | CPU threads per CPU process                    |
| `-ntg, --threads-gpu-training`  | CPU threads per GPU training process           |
| `-ngi, --threads-gpu-inference` | CPU threads per GPU inference process          |
| `-mc, --memory-cpu`             | Memory (MB) per CPU process                    |
| `-mgt, --memory-gpu-training`   | Memory (MB) per GPU training process           |
| `-mgi, --memory-gpu-inference`  | Memory (MB) per GPU inference process          |
| `--prepare`                     | Only generate scripts, don’t submit            |
| `--dry-run`                     | Run in dev mode with reduced resources         |

### What Happens

* Loads cluster/job config from [slurm.toml](slurm.toml)
* Applies CLI overrides
* Checks the current hostname matches the expected hostname (prevents accidental submission)
* Calculates CPU/GPU process counts based on node resources
* Creates a job directory named after `JOBNAME`
* Copies essential config files ([config.toml](config.toml), [constants.json](constants.json))
* Generates SLURM batch scripts using Jinja2 templates
* Submits job unless run with `--prepare`

### Example

```bash
python submit.py my_experiment -t 24 -n 2 -q gpu --conda_env rlz_env -ntg 4 -ngi 2 -mc 2000
```

Creates and submits a 24-hour job called `my_experiment` on 2 GPU nodes using the `rlz_env` conda environment with specified resource parameters.

---

If you'd like, I can also help you create a **usage snippet** or **example script** for easy copy-pasting!

## Development

This project uses poetry for managing dependencies for Python 3.10. To use use the environment run
```bash
poetry install --with mpi
poetry env activate
```

To install pre-commit hooks use
```bash
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

To run tests:
```bash
poetry run pytest
```

To get test coverage:
```bash
poetry run coverage run -m pytest
```
and for the interactive html:
```bash
poetry run coverage html
```

For testing the current implementation the main provides an entry point `slurm-test` that emulates a slurm exection withjout relying on the slurm environments.
The Usage is:

```bash
mpiexec -n 4 poetry run python src/main.py slurm-test --num-gpus=1 -agents-per-observer=1
```

Currently there remains the past version of execution with adative choice of processes but will be removed in coming versions.


## More Details

For details regarding the Considerations, Planning and more:

### [Summary](resources/Summary.md)
