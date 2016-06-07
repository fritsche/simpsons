# Data Augmentation

for i in 90 180 270; do	
	rm -rf data/$i/
	mkdir -p data/$i/
	for szFile in data/TrainOriginal/*.bmp ; do 
		output=${szFile##*/}
		convert $szFile -rotate $i data/$i/${output%.*}_$i.bmp
		echo data/$i/${output%.*}_$i.bmp
	done
done


rm -rf data/flop/
mkdir -p data/flop/

for szFile in data/TrainOriginal/*.bmp ; do 
	output=${szFile##*/}
	convert $szFile -flop data/flop/${output%.*}_flop.bmp
	echo data/flop/${output%.*}_flop.bmp
done

rm -rf data/Train
mkdir -p data/Train

cp -rv data/TrainOriginal/* data/Train/

for i in 90 180 270; do	
	cp -rv data/$i/* data/Train/
done

cp -rv data/flop/* data/Train/
