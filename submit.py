import os
import shutil
import toml
import click
import socket
from jinja2 import Template
import subprocess


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

    # generate the multi_prog.conf dynamically
    generate_multi_prog_conf(jobname, gpu_train, gpu_inf, cpu_procs)

    # render the SLURM script from template
    render_slurm_script(jobname, cfg)

    # submit job unless prepare
    if not prepare:
        submit_job(jobname)
    else:
        click.echo(f"--prepare: scripts are in {jobname}/")


def generate_multi_prog_conf(jobname, gpu_train, gpu_inf, cpu_procs):
    """Generate the multi_prog.conf file dynamically."""
    multi_prog_file = os.path.join(jobname, "multi_prog.conf")
    
    # Write the tasks for training, inference, and worker
    with open(multi_prog_file, "w") as f:
        # Task 0: Training process
        f.write("0 python src/main.py slurm-shared-gpu -apo=1\n")
        
        # Task 1 to 2: Inference processes
        for i in range(1, gpu_inf + 1):
            f.write(f"{i} python src/main.py slurm-shared-gpu -apo=1\n")
        
        # Task 3 to N: CPU worker processes
        task_id = gpu_inf + 1
        for i in range(task_id, task_id + cpu_procs):
            f.write(f"{i} python src/main.py  slurm-shared-gpu -apo=1\n")

    return multi_prog_file


def render_slurm_script(jobname, cfg):
    """Render the SLURM job script using Jinja2."""
    # Load SLURM template
    tmpl = Template(open("sbatch_template.j2").read())

    # Define context for the template
    ctx = {
        "jobname": jobname,
        "time": cfg["prod"]["time"],
        "queue": cfg["general"]["queue"],
        "conda_env": cfg["general"]["conda_env"],
        "gpu_train_proc": cfg["prod"]["gpu"]["training_processes"],
        "gpu_threads_t": cfg["prod"]["gpu"]["training"]["threads_per_gpu"],
        "gpu_mem_t": cfg["prod"]["gpu"]["training"]["mem_per_process"],
        "gpu_inference_proc": cfg["prod"]["gpu"]["inference_processes"],
        "gpu_threads_i": cfg["prod"]["gpu"]["inference"]["threads_per_gpu"],
        "gpu_mem_i": cfg["prod"]["gpu"]["inference"]["mem_per_process"],
        "cpu_processes": cfg["prod"]["worker_cpus_total"],
        "cpu_threads": cfg["prod"]["cpu"]["threads_per_process"],
        "cpu_memory": cfg["prod"]["cpu"]["mem_per_process"],
        "agents_per_observer": cfg["prod"]["cpu"]["agents_per_observer"],
    }

    # Render the SLURM script
    out = tmpl.render(**ctx)

    # Write the rendered script to the job directory
    with open(os.path.join(jobname, "submit.slurm"), "w") as f:
        f.write(out)


def submit_job(jobname):
    """Submit the SLURM job."""
    slurm_script = os.path.join(jobname, "submit.slurm")
    subprocess.run(f"sbatch {slurm_script}", shell=True)


if __name__ == "__main__":
    main()

