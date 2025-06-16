import os
import shutil
import toml
import click
import socket
from jinja2 import Template


def check_hostname(cfg):
    """Enforce that user checks the Hostname ie Slurm config before submitting."""
    expected = cfg.get("general", {}).get("hostname")
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
@click.option("-apo", "--agents-per-observer", default=None, type=int)
@click.option("-ntc", "--threads-cpu-process", default=None, type=int)
@click.option("-ntg", "--threads-gpu-training", default=None, type=int)
@click.option("-ngi", "--threads-gpu-inference", default=None, type=int)
@click.option("-mc", "--memory-cpu", default=None, type=int, help="MB per CPU Process")
@click.option("-mgt", "--memory-gpu-training", default=None, type=int)
@click.option("-mgi", "--memory-gpu-inference", default=None, type=int)
@click.option("--prepare", is_flag=True, help="Only prepare scripts, don't submit")
@click.option("--dry-run", is_flag=True, help="Run without full node requirement")
def main(
    jobname,
    time,
    nodes,
    queue,
    conda_env,
    agents_per_observer,
    threads_cpu_process,
    threads_gpu_training,
    threads_gpu_inference,
    memory_cpu,
    memory_gpu_training,
    memory_gpu_inference,
    prepare,
    dry_run,
):
    """Generate slurm run config with separate GPU allocations for training and inference."""
    cfg = toml.load(open("slurm.toml"))

    check_hostname(cfg)

    # Override sections
    overrides = {
        "general": {"queue": queue, "conda_env": conda_env},
        "prod": {"time": time, "nodes": nodes},
        "prod.cpu": {
            "threads_per_process": threads_cpu_process,
            "mem_per_process": memory_cpu,
        },
        "prod.gpu.training": {
            "threads_per_gpu": threads_gpu_training,
            "mem_per_process": memory_gpu_training,
        },
        "prod.gpu.inference": {
            "threads_per_gpu": threads_gpu_inference,
            "mem_per_process": memory_gpu_inference,
            "agents_per_observer": agents_per_observer,
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

    # General mode and dry-run
    cfg.setdefault("general", {})
    cfg["general"]["mode"] = "dev" if dry_run else cfg["general"].get("mode", "prod")
    cfg["general"]["prepare"] = prepare
    mode = cfg["general"]["mode"]

    # Compute processes per role
    gpus_per_node = cfg["general"]["gpus_per_node"]
    gpu_train_proc = cfg["prod"]["gpu"]["training_processes"] = cfg["prod"]["gpu"].get(
        "training_processes", gpus_per_node
    )
    gpu_inf_proc = cfg["prod"]["gpu"]["inference_processes"] = cfg["prod"]["gpu"].get(
        "inference_processes", gpus_per_node
    )

    gpu_threads_t = cfg["prod"]["gpu"]["training"]["threads_per_gpu"]
    gpu_threads_i = cfg["prod"]["gpu"]["inference"]["threads_per_gpu"]
    gpu_mem_t = cfg["prod"]["gpu"]["training"]["mem_per_process"]
    gpu_mem_i = cfg["prod"]["gpu"]["inference"]["mem_per_process"]
    threads_per_cpu = cfg["prod"]["cpu"]["threads_per_process"]
    gpus_per_node = cfg["general"]["gpus_per_node"]

    if mode == "dev":
        cfg["prod"]["time"]

    # CPU processes

    if mode == "prod":
        max_cpus = cfg["general"]["cpus_per_node"]
    elif mode == "dev":
        max_cpus = cfg["dev"].get("num_processes", cfg["general"]["cpus_per_node"])
    else:
        raise ValueError(f"{mode} is not a valid general.mode")

    cpu_avail_node0 = -(
        gpu_train_proc * gpu_threads_t + (gpus_per_node - 1) * gpu_threads_i
    )
    cpu_avail_noden = max_cpus - (gpus_per_node * gpu_threads_i)
    cpu_processes_node0 = cpu_avail_node0 // threads_per_cpu
    cpu_processes_noden = cpu_avail_noden // threads_per_cpu
    cfg["prod"]["cpu"]["cpu_processes_node0"] = max(1, cpu_processes_node0)
    cfg["prod"]["cpu"]["cpu_processes_noden"] = max(1, cpu_processes_noden)

    # Prepare output directory
    print(f"Preparing Job directory: {jobname}")
    os.makedirs(jobname, exist_ok=True)
    for fname in ("config.toml", "constants.json"):
        target = os.path.join(jobname, fname)
        if not os.path.exists(target):
            click.echo(f"Copying {fname} → {jobname}/")
            shutil.copy(fname, target)

    # Render sbatch header
    tmpl = Template(open("sbatch_template.j2").read())
    header = tmpl.render(
        jobname_prefix="rlz",
        jobname=jobname,
        test_only=False,
        gpus_per_node=cfg["general"]["gpus_per_node"],
        conda_env=cfg["general"]["conda_env"],
        time=cfg["prod"]["time"],
        nodes=cfg["prod"]["nodes"],
        queue=cfg["general"].get("queue", "std"),
        gpu_train_proc=gpu_train_proc,
        gpu_threads_t=gpu_threads_t,
        gpu_mem_t=gpu_mem_t,
        gpu_inf_proc=gpu_inf_proc,
        gpu_threads_i=gpu_threads_i,
        gpu_mem_i=gpu_mem_i,
        cpu_processes_node0=cfg["prod"]["cpu"]["cpu_processes_node0"],
        cpu_processes_noden=cfg["prod"]["cpu"]["cpu_processes_noden"],
        cpu_threads=cfg["prod"]["cpu"]["threads_per_process"],
        cpu_memory=cfg["prod"]["cpu"]["mem_per_process"],
        agents_per_observer=cfg["prod"]["cpu"]["agents_per_observer"],
    )
    run_slurm_path = os.path.join(jobname, f"run.{jobname}.slurm")
    with open(run_slurm_path, "w") as f:
        f.write(header)
    print("Done writing run script.")

    est_header = tmpl.render(
        jobname_prefix="rlz",
        jobname=jobname,
        test_only=True,
        gpus_per_node=cfg["general"]["gpus_per_node"],
        conda_env=cfg["general"]["conda_env"],
        time=cfg["prod"]["time"],
        nodes=cfg["prod"]["nodes"],
        queue=cfg["general"].get("queue", "std"),
        gpu_train_proc=gpu_train_proc,
        gpu_threads_t=gpu_threads_t,
        gpu_mem_t=gpu_mem_t,
        gpu_inf_proc=gpu_inf_proc,
        gpu_threads_i=gpu_threads_i,
        gpu_mem_i=gpu_mem_i,
        cpu_processes_node0=cfg["prod"]["cpu"]["cpu_processes_node0"],
        cpu_processes_noden=cfg["prod"]["cpu"]["cpu_processes_noden"],
        cpu_threads=cfg["prod"]["cpu"]["threads_per_process"],
        cpu_memory=cfg["prod"]["cpu"]["mem_per_process"],
        agents_per_observer=cfg["prod"]["cpu"]["agents_per_observer"],
    )
    est_path = os.path.join(jobname, f"estimate.{jobname}.slurm")
    with open(est_path, "w") as f_est:
        f_est.write(est_header)
    print("Done writing estimation script.")

    # --- Auto-submit the appropriate script with feedback ---
    if not prepare:
        click.echo(f"Starting Job: {jobname}")
        click.echo("Currently estimating Batch execution time:")
        os.system(f"sbatch {est_path}")
        os.system(f"sbatch {run_slurm_path}")
    else:
        click.echo(
            f"Not running submission script as run with --prepare flag.\nCan be run manually with:\n\n    sbatch run.{jobname}.slurm\n\nCurrently estimating Batch execution time using {jobname}/estimate.{jobname}.slurm:"
        )
        os.system(f"sbatch {est_path}")


if __name__ == "__main__":
    main()
