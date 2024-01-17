
export FILEPATH="/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5"
export RESULTS_PATH="./"
export PREPROCESSING=True
export N_SAMPLES=5
export DEVICE="gpu"

for n in {1..4}
do
    for c in {2..4}
    do
        # "normal" parallel K-means
        export METHOD="km"
        for i in "++" "random" "batchparallel"
        do
            mpirun -n $n python ~/heat/examples/cluster/demo_batchparallelclustering.py --filepath $FILEPATH --results_path $RESULTS_PATH --n_clusters $c --n_samples $N_SAMPLES --device $DEVICE --init $i --method $METHOD --preprocessing $PREPROCESSING
        done

        # batch parallel k-Means
        export METHOD="bpkm"
        export INIT="++"
        for m in 0 2
        do
            mpirun -n $n python ~/heat/examples/cluster/demo_batchparallelclustering.py --filepath $FILEPATH --results_path $RESULTS_PATH --n_clusters $c --n_samples $N_SAMPLES --device $DEVICE --init $INIT --method $METHOD --preprocessing $PREPROCESSING --n_merges $m
        done
    done
done
