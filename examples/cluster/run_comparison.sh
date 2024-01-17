
export FILEPATH="/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5"
export RESULTS_PATH="./"
export PREPROCESSING=True
export N_SAMPLES=5


for c in {2..4}
do
    for i in "++" "random"
    do
        # "normal" parallel K-means
        export METHOD="km"
        python ~/heat/examples/cluster/comparison_sklearn.py --filepath $FILEPATH --results_path $RESULTS_PATH --n_clusters $c --n_samples $N_SAMPLES --init $i --method $METHOD --preprocessing $PREPROCESSING

        # batch parallel k-Means
        export METHOD="mbkm"
        for b in 100 1024
        do
            python ~/heat/examples/cluster/comparison_sklearn.py --filepath $FILEPATH --results_path $RESULTS_PATH --n_clusters $c --n_samples $N_SAMPLES --init $i --method $METHOD --preprocessing $PREPROCESSING --batch_size $b
        done
    done
done
