#!/usr/bin/env bash

OUT=bm.out

for i in 10 25 50 100 200 500 1000 1500 2000; do 
	for j in `seq 1 10`; do
		echo "k = $i, run $j:" >> $OUT;
		{ time ./kmeans -k $i; } >> $OUT 2>&1;
		echo -e "\n\n" >> $OUT; 
	done
done
