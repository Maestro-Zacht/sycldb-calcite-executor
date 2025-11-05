#!/bin/bash

query=$1

if [ $query == "q11" ] || [ $query == "q12" ] || [ $query == "q13" ]; then
    cat "./${query}.res"
elif [ $query == "q21" ] || [ $query == "q22" ] || [ $query == "q23" ]; then
    sort "./${query}.res" -nk 2 -nk 3
elif [ $query == "q31" ] || [ $query == "q32" ] || [ $query == "q33" ] || [ $query == "q34" ]; then
    sort "./${query}.res" -nk 3 -nrk 4
elif [ $query == "q41" ]; then
    sort "./${query}.res" -nk 1 -nk 2
elif [ $query == "q42" ] || [ $query == "q43" ]; then
    sort "./${query}.res" -nk 1 -nk 2 -nk 3
else
    echo "Query not recognized for sorting."
fi
