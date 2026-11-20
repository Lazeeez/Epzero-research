#!/usr/bin/env bash
# Read-only cluster/account inventory. No installations, downloads, or compute on a login node.
set -euo pipefail
hostname
id -un
uname -sr
command -v sbatch srun sinfo python3
sinfo -h -o '%P|%a|%l|%D|%G|%f'
squeue -h -u "$USER" -o '%i|%P|%j|%T|%M|%D|%R'
if command -v sacctmgr >/dev/null; then
    sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS || true
fi
if [ -d "$HOME/scratch" ]; then
    readlink -f "$HOME/scratch"
    df -h "$HOME/scratch"
fi
if command -v quota >/dev/null; then
    quota -s || true
fi
