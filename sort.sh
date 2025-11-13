#!/bin/bash

query=$1

if [ $query == "q11" ] || [ $query == "q12" ] || [ $query == "q13" ]; then
    :
elif [ $query == "q21" ] || [ $query == "q22" ] || [ $query == "q23" ]; then
    sort "./${query}.res" -k2,2n -k3,3n -o ./${query}.res
elif [ $query == "q31" ] || [ $query == "q32" ] || [ $query == "q33" ] || [ $query == "q34" ]; then
    sort "./${query}.res" -k3,3n -k4,4nr -o ./${query}.res
elif [ $query == "q41" ]; then
    sort "./${query}.res" -k1,1n -k2,2n -o ./${query}.res
elif [ $query == "q42" ] || [ $query == "q43" ]; then
    sort "./${query}.res" -k1,1n -k2,2n -k3,3n -o ./${query}.res
else
    echo "Query not recognized for sorting."
fi
