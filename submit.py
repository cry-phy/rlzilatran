import os
import shutil
import toml
import click
import socket
from jinja2 import Template


def check_hostname(cfg):
    expected = cfg["general"].get("hostname")
    actual = socket.gethostname()
    if not expected:
        raise ValueError("Missing general.hostname in slurm.toml")
    if expected not in actual:
        raise RuntimeError(f"Hostname mismatch: expected '{expected}', got '{actual}'")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("jobname")
@click.option("-t", "--time", type=int, help="Hours")
@click.option("-q", "--queue", type=str, help="SLURM partition")
@click.option("--conda-env", type=str, help="Conda environment")
@click.option("-wp", "--worker-cpus", type=int, help="Total CPU worker processes")
@click.option("-apo", "--agents-per-observer", type=int)
@click.option("-ntg", "--threads-gpu-train", type=int)
@click.option("-ngi", "--threads-gpu-infer", type=int)
@click.option("-mgt", "--memory-gpu-train", type=int)
@click.option("-mgi", "--memory-gpu-infer", type=int)
@click.option("--prepare", is_flag=True, help="Only prepare scripts, don't sbatch")
@click.option("--dry-run", is_flag=True, help="Use dev settings instead of prod")
def main(
    jobname,
    time,
    queue,
    conda_env,
    worker_cpus,
    agents_per_observer,
    threads_gpu_train,
    threads_gpu_infer,
    memory_gpu_train,
    memory_gpu_infer,
    prepare,
    dry_run,
):
    # load config
    cfg = toml.load("slurm.toml")
    check_hostname(cfg)

    # apply overrides
    def apply(section, key, val):
        if val is not None:
            sec = cfg
            for part in section.split("."):
                sec = sec.setdefault(part, {})
            sec[key] = val

    apply("general", "queue", queue)
    apply("general", "conda_env", conda_env)
    apply("prod", "time", f"{time:02d}:00:00" if time else None)
    apply("prod", "worker_cpus_total", worker_cpus)
    apply("prod.cpu", "agents_per_observer", agents_per_observer)
    apply("prod.gpu.training", "threads_per_gpu", threads_gpu_train)
    apply("prod.gpu.training", "mem_per_process", memory_gpu_train)
    apply("prod.gpu.inference", "threads_per_gpu", threads_gpu_infer)
    apply("prod.gpu.inference", "mem_per_process", memory_gpu_infer)

    # set mode
    cfg["general"]["mode"] = "dev" if dry_run else "prod"
    cfg["general"]["prepare"] = prepare

    # extract settings
    gpn = cfg["general"]["gpus_per_node"]
    total_workers = cfg["prod"]["worker_cpus_total"]

    # GPU processes
    gpu_train = cfg["prod"]["gpu"].get("training_processes", gpn)
    gpu_inf = cfg["prod"]["gpu"].get("inference_processes", gpn)
    gt = cfg["prod"]["gpu"]["training"]["threads_per_gpu"]
    gi = cfg["prod"]["gpu"]["inference"]["threads_per_gpu"]
    mt = cfg["prod"]["gpu"]["training"]["mem_per_process"]
    mi = cfg["prod"]["gpu"]["inference"]["mem_per_process"]

    # CPU workers (flat total)
    cpu_procs = max(1, total_workers)

    # prepare job directory
    os.makedirs(jobname, exist_ok=True)
    for fn in ("config.toml", "constants.json"):
        if not os.path.exists(os.path.join(jobname, fn)):
            shutil.copy(fn, jobname)

    # render templates
    tmpl = Template(open("sbatch_template.j2").read())
    ctx = {
        "jobname_prefix": "rlz",
        "jobname": jobname,
        "test_only": False,
        "time": cfg["prod"]["time"],
        "queue": cfg["general"]["queue"],
        "conda_env": cfg["general"]["conda_env"],
        "gpu_train_proc": gpu_train,
        "gpu_threads_t": gt,
        "gpu_mem_t": mt,
        "gpu_inference_proc": gpu_inf,
        "gpu_threads_i": gi,
        "gpu_mem_i": mi,
        "cpu_processes": cpu_procs,
        "cpu_threads": cfg["prod"]["cpu"]["threads_per_process"],
        "cpu_memory": cfg["prod"]["cpu"]["mem_per_process"],
        "agents_per_observer": cfg["prod"]["cpu"]["agents_per_observer"],
    }

    for tag, flag in [("run", False), ("estimate", True)]:
        ctx["test_only"] = flag
        out = tmpl.render(**ctx)
        path = os.path.join(jobname, f"{tag}.{jobname}.slurm")
        with open(path, "w") as f:
            f.write(out)
        click.echo(f"Wrote {path}")

    # submit unless prepare
    if not prepare:
        click.echo("Estimating time…")
        os.system(f"sbatch {jobname}/estimate.{jobname}.slurm")
        click.echo("Submitting job…")
        os.system(f"sbatch {jobname}/run.{jobname}.slurm")
    else:
        click.echo(f"--prepare: scripts are in {jobname}/")
        click.echo("Estimating time…")
        os.system(f"sbatch {jobname}/estimate.{jobname}.slurm")


if __name__ == "__main__":
    main()
