#!/bin/bash -x

for i in {0..2}
do
    numactl --membind "$i" ./client ./queries/transformed/q11.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q12.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q13.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q21.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q22.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q23.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q31.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q32.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q33.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q34.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q41.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q42.sql "$i"
    numactl --membind "$i" ./client ./queries/transformed/q43.sql "$i"
done