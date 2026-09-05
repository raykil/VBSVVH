MANAGER_NAME=jkil-makeDF_2018_data
N_WORKERS=40
WRAPPER=/users/jkil/with_oasis_certs
CORES=12
MEMORY_GB=10
DISK_GB=100
PONCHO_ENV=lfv-coffea-env.tar.gz

# Ignore SIGPIPE so broken pipes don't terminate the script.
trap '' PIPE

while [ $# -gt 0 ]; do
    case "$1" in
        -M)
            [ $# -ge 2 ] || exit 2
            MANAGER_NAME="$2"
            shift 2
            ;;
        --manager-name)
            [ $# -ge 2 ] || exit 2
            MANAGER_NAME="$2"
            shift 2
            ;;
        --cores)
            [ $# -ge 2 ] || exit 2
            CORES="$2"
            shift 2
            ;;
        --memory)
            [ $# -ge 2 ] || exit 2
            MEMORY_GB="$2"
            shift 2
            ;;
        --disk)
            [ $# -ge 2 ] || exit 2
            DISK_GB="$2"
            shift 2
            ;;
        --workers)
            [ $# -ge 2 ] || exit 2
            N_WORKERS="$2"
            shift 2
            ;;
        --wrapper)
            [ $# -ge 2 ] || exit 2
            WRAPPER="$2"
            shift 2
            ;;
        --poncho-env)
            [ $# -ge 2 ] || exit 2
            PONCHO_ENV="$2"
            shift 2
            ;;
        -*)
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

MEMORY=$((MEMORY_GB*1024))
DISK=$((DISK_GB*1024))

vine_cmd=(
    vine_factory
    -T condor
    --scratch-dir /groups/cjessop/users/jkil
    --poncho-env "$PONCHO_ENV"
    --wrapper "$WRAPPER"
    # --condor-requirements '((has_vast))'
    --manager-name "$MANAGER_NAME"
    --min-workers "$N_WORKERS"
    --max-workers "$N_WORKERS"
    --workers-per-cycle "$N_WORKERS"
    --cores "$CORES"
    --memory "$MEMORY"
    --disk "$DISK"
    --timeout 36000
)

"${vine_cmd[@]}"

# condor_ssh_to_job 127709
# condor_q -better-analyze 127715
# after the jobs are done, I should shut down the workers by doing ctrl+c.
# 
# ps aux | grep jkil | grep run_processor.py # to see if there are zombie processes
# https://github.com/cooperative-computing-lab/cctools/blob/master/batch_job/src/vine_factory.c
# https://cctools.readthedocs.io/en/latest/man_pages/vine_factory/

# Number of workers available with memory higher than requested:
# condor_status -constraint 'SlotType != "Dedicated" && State=="Unclaimed" && Memory >= (52*1024)' -af Name | wc -l