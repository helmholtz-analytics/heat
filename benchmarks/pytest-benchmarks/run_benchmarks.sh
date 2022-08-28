#! /usr/bin/bash
var=$(pwd)/'aggregate.py'
repo=$(pwd)/'results/data/'
for d in */; do
    if [ "$d" = "results/" ];then
        continue
    fi
    cd "$d"
    pytest --benchmark-json output.json
    python $var
    rm output.json
    mkdir $repo"$d"
    cp aggregate.json $repo"$d"
    cd ..
done
cd "$repo"
python buildlist.py . > list.json
rsync -rtv ${HOME}/fork/heat/benchmarks/pytest-benchmarks/results/ ${HOME}/fork/heat-data/
cd ${HOME}/fork/heat-data/
git commit -am "heat benchmark"
git push