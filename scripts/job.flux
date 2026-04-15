#!/bin/bash
#flux: -N 1
#flux: -t 5m
#flux: -q pdebug
#flux: -l

flux env

export HSA_XNACK=1
rm -rf fp64_monitor_output
flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install-6.4.3/bin/fp64_monitor.sh --quiet --interval 100 --counters fp64 --dir fp64 -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install-6.4.3/bin/fp64_monitor.sh --quiet --interval 100 --counters fp32 --dir fp32 -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install-6.4.3/bin/fp64_monitor.sh --quiet --interval 100 --counters int --dir int -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install-6.4.3/bin/fp64_monitor.sh --quiet --interval 100 --counters salu --dir salu -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#export FP64_MONITOR_INTERVAL=100
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install-6.4.3/bin/fp64_monitor.sh ./install-6.4.3/bin/test_mfma_direct

