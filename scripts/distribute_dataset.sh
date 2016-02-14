#!/bin/bash

# Sequentially check sax* directories for each patient
# If directory contains more than one sequence, then splits it into separete directories

start_dir=$(pwd)

cd $1

for i in $(ls -1); do
	cd $i/study;
	for s in $(ls --directory sax_*); do
		SAX=${s//sax_}
		count=$(ls -1 ${s}/*.dcm | wc -l)
		echo $i $SAX $count;
		
		if [ $count -ge 31 ]; then
			for img in $(ls -1 sax_${SAX}/*.dcm); do
				img=${img//sax_${SAX}\/}
				<<< $(echo ${img} | tr '-' ' ') read skip skip index new_sax
				new_sax_index=${new_sax%.dcm}
				new_sax_index=${new_sax_index:2:2}
				
				new_sax_index=sax_${SAX}${new_sax_index}
				mkdir -p ${new_sax_index}
				mv sax_${SAX}/${img} ${new_sax_index}/${img}
				echo mv sax_${SAX}/${img} ${new_sax_index}/${img}
			done
		fi
		
	done;
	cd ../..;
done;

cd start_dir