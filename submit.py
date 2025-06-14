import os
import shutil
import toml
import click
import socket
from jinja2 import Template


def check_hostname(cfg):
    """Enforce that user checks the Hostname ie Slurm config before submitting."""
    expected = cfg.get("system", {}).get("hostname")
    actual = socket.gethostname()

    if expected is None:
        raise ValueError("Expected hostname not set in [system] section of slurm.toml")

    if expected not in actual:
        raise RuntimeError(
            f"Hostname mismatch: expected '{expected}', but got '{actual}'"
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("jobname")
@click.option("-t", "--time", default=None, type=int, help="Hours")
@click.option("-n", "--nodes", default=None, type=int, help="Number of filled nodes")
@click.option("-q", "--queue", default=None, type=str, help="SLURM partition")
@click.option("--conda_env", default=None, type=str, help="Conda environment to use.")
@click.option("-ntc", "--threads-cpu-process", default=None, type=int)
@click.option("-ntg", "--threads-gpu-process", default=None, type=int)
@click.option("-mc", "--memory-cpu", default=None, type=int, help="MB per CPU Process")
@click.option("-mg", "--memory-gpu", default=None, type=int, help="MB per GPU Process")
@click.option("--prepare", is_flag=True, help="Only prepare scripts, don't submit")
@click.option("--dry-run", is_flag=True, help="Run without full node requirement")
def main(
    jobname,
    time,
    nodes,
    queue,
    conda_env,
    threads_cpu_process,
    threads_gpu_process,
    memory_cpu,
    memory_gpu,
    prepare,
    dry_run,
):
    """Generate slurm run config."""
    cfg = toml.load(open("slurm.toml"))

    overrides = {
        "general": {"queue": queue, "conda_env": conda_env},
        "prod": {"time": time, "nodes": nodes},
        "prod.cpu": {
            "threads_per_process": threads_cpu_process,
            "mem_per_process": memory_cpu,
        },
        "prod.gpu": {
            "threads_per_gpu": threads_gpu_process,
            "mem_per_process": memory_gpu,
        },
    }

    # Apply overrides
    for section, values in overrides.items():
        parts = section.split(".")
        subcfg = cfg
        for part in parts:
            if part not in subcfg:
                subcfg[part] = {}
            subcfg = subcfg[part]
        for key, value in values.items():
            if value is not None:
                subcfg[key] = value

    # Set general mode
    cfg.setdefault("general", {})
    cfg["general"]["mode"] = "dev" if dry_run else cfg["general"]["mode"]
    cfg["general"]["prepare"] = prepare

    mode = cfg["general"]["mode"]

    gpus_per_node = cfg["general"]["gpus_per_node"]
    gpu_threads = cfg["prod"]["gpu"]["threads_per_gpu"]
    gpu_processes = gpus_per_node

    if mode == "prod":
        cpu_avail = cfg["general"]["cpus_per_node"] - gpu_processes * gpu_threads
        threads_per_cpu = cfg["prod"]["cpu"]["threads_per_process"]
        cpu_processes = cpu_avail // threads_per_cpu
    elif mode == "dev":
        dev_total = cfg["prod"].get("num_processes", 4)
        cpu_processes = max(1, dev_total - gpu_processes)
    else:
        raise ValueError(f"{mode} is not a valid general.mode")

    cfg["prod"]["cpu"]["cpu_processes"] = cpu_processes

    print(f"Preparing Job directory: {jobname}")
    os.makedirs(jobname, exist_ok=True)
    for fname in ("config.toml", "constants.json"):
        target = os.path.join(jobname, fname)
        if not os.path.exists(target):
            click.echo(f"Copying {fname} → {jobname}/")
            shutil.copy(fname, target)

    tmpl = Template(open("sbatch_header.j2").read())
    header = tmpl.render(
        jobname_prefix="rlz",
        jobname=jobname,
        test_only=prepare,
        conda_env=cfg["general"]["conda_env"],
        time=cfg["prod"]["time"],
        nodes=cfg["prod"]["nodes"],
        queue=cfg["general"].get("queue", "std"),
        gpu_processes=gpu_processes,
        gpu_threads=cfg["prod"]["gpu"]["threads_per_gpu"],
        gpu_memory=cfg["prod"]["gpu"]["mem_per_process"],
        cpu_processes=cfg["prod"]["cpu"]["cpu_processes"],
        cpu_threads=cfg["prod"]["cpu"]["threads_per_process"],
        cpu_memory=cfg["prod"]["cpu"]["mem_per_process"],
    )
    with open(os.path.join(jobname, "submit.slurm"), "w") as f:
        f.write(header)
    print("Done writing run script")


if __name__ == "__main__":
    main()
