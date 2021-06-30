#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function shutdown {
  sudo shutdown -h now
}

trap shutdown EXIT
singularity exec --nv container.img python bin/aiCompetition.py  -t 3600 --topK 5 --arity 4 --maximumFrontier 5 -i 10 -R 3600 -RS 5000 --biasOptimal --contextual --mask  -r 0. 2>&1 | tee log.txt
