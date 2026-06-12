#!/bin/bash
#SBATCH --job-name=cache16
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --qos=publicgrp-low-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --array=0-15
#SBATCH --output=/quobyte/proteomics-grp/brett/glendon/cache16_%a.log
source /quobyte/proteomics-grp/brett/miniforge3/etc/profile.d/conda.sh
conda activate /quobyte/proteomics-grp/brett/envs/alphadia2
export TMPDIR=/quobyte/proteomics-grp/brett/tmp
mapfile -t FILES < /quobyte/proteomics-grp/brett/glendon/fragpipe_bigdog_diatracer_16/subset16.txt
D="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASE=$(basename "$D" .d)
python /quobyte/proteomics-grp/brett/glendon/cache_one.py "$D" "/quobyte/proteomics-grp/brett/glendon/cache16/$BASE"
