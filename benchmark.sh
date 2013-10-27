#!/usr/bin/env bash

OUT=bm.out

CONFIG=("--cpu" "--blocksize 256")

:>$OUT

for i in 10 25 50 100 200 500 1000 1500; do 
	for cfg in $CONFIG; do
		echo "k = $i, $cfg" >> $OUT;
		for j in `seq 1 10`; do
			#echo "k = $i, run $j:" >> $OUT;
			{ ./kmeans -k $i $cfg; } >> $OUT 2>&1;
				#echo -e "\n\n" >> $OUT; 
			done
			echo -e "\n" >> $OUT; 
		done
	done
