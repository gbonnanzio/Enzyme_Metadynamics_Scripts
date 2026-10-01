#!/bin/bash
jid0=($(sbatch --time=00:30:00 --account=p30041 --partition=gengpu --constraint=sxm --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --mem=10G --job-name="min" --output=min.txt amber_min.sh))

echo "jid0 ${jid0[-1]}" >> slurm_ids

jid1=($(sbatch --dependency=afterok:${jid0[-1]} --time=28:00:00 --account=p30041 --partition=gengpu --gres=gpu:h100:1 --nodes=1 --ntasks-per-node=1 --mem-per-cpu=10G --job-name="DMC" --output=run.txt --export=DEPENDENTJOB=${jid0[-1]} gpu_amber_job.sh))

echo "jid1 ${jid1[-1]}" >> slurm_ids

jid2=($(sbatch --dependency=afterok:${jid1[-1]} --time=4:00:00 --account=p30041 --partition=short --nodes=1 --ntasks-per-node=10 --mem-per-cpu=5G --job-name="D-dG" --output=mmgbsa.txt --export=DEPENDENTJOB=${jid1[-1]} amber_mmgbsa.sh))

echo "jid2 ${jid2[-1]}" >> slurm_ids
