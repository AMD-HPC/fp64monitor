#!/bin/bash
#flux: -N 1
#flux: -t 5m
#flux: -q pdebug
#flux: -l

flux env
jid=`flux getattr jobid`

export HSA_XNACK=1

# Insert the monitor between flux arguments and executable name / arguments, ending with "--"
# Customize the output directory with flux job ID and step ID because tool will overwrite the directory otherwise
flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 \
./install/bin/fp64_monitor.sh --quiet --interval 100 --counters fp64 --rocm 6.4.3 --dir fp64_${jid}_{{id}} -- \
./install-6.4.3/bin/stream_triad_fp64 16777216 1000

#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install/bin/fp64_monitor.sh --quiet --interval 100 --counters fp32 --rocm 6.4.3 --dir fp32_${jid}_{{id}} -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install/bin/fp64_monitor.sh --quiet --interval 100 --counters int --rocm 6.4.3 --dir int_${jid}_{{id}} -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install/bin/fp64_monitor.sh --quiet --interval 100 --counters salu --rocm 6.4.3 --dir salu_${jid}_{{id}} -- ./install-6.4.3/bin/stream_triad_fp64 16777216 1000
#export FP64_MONITOR_INTERVAL=100
#flux run -N 1 -n 4 -c 16 -g 1 -x --unbuffered --verbose --setopt=mpibind=verbose:1 ./install/bin/fp64_monitor.sh --rocm 6.4.3  -- ./install-6.4.3/bin/test_mfma_direct

