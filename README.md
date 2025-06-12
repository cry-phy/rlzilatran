# RLZilatran

## Slurm Usage

To get the project running using the Slurm Workload manager and conda first ensure that poetry does not currently have a venv (In case remove it.). Then create a conda environment (default is *rlzilatran*) using ac compatible python version. Then simply let poetry install to said env.
```bash
poetry config virtualenvs.create false --local
conda activate rlzilatran # or your conda environment
poetry install
```
Usage is then centralized to [Slurm script](SlurmExecute.slurm).

## Development

This project uses poetry for managing dependencies for Python 3.10. To use use the environment run
```bash
poetry install
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

## More Details

For details regarding the Considerations, Planning and more:

### [Summary](resources/Summary.md)
