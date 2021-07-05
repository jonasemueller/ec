#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function shutdown {
  sudo shutdown -h now
}

trap shutdown EXIT

# We use the presence of the "nvidia-container-cli" command to determine whether
# to run Singularity with CUDA. The "nvidia-container-cli" approach doesn't
# discover the presence of CUDA in all possible scenarios so it can also be
# manually activated by setting the environment variable.
if command -v nvidia-container-cli &> /dev/null
then
  export GPU="--nv"
fi

singularity exec ${GPU:+"$GPU"} container.img python bin/aiCompetition.py \
  --biasOptimal --contextual 2>&1 | tee log.txt
