#!/bin/bash

features_list=(color_histogram )
classifiers=( svm )

# features_list=(color_histogram hu_moments histogram_of_oriented_gradients)
# classifiers=( svm lda knn )

for features in "${features_list[@]}"
do
	for classifier in "${classifiers[@]}"
	do
		
		echo "Classifier: $classifier"
		echo "Feature extraction: $features" 

		python feature_extraction.py data/flop/ $features
		echo $( wc -l < data/features.txt )  $( awk '{print NF}' data/features.txt | head -n 1 ) > data/train.txt
		paste data/features.txt data/classes.txt >> data/train.txt

		python feature_extraction.py data/Valid/ $features
		echo $( wc -l < data/features.txt )  $( awk '{print NF}' data/features.txt | head -n 1 ) > data/test.txt
		paste data/features.txt data/classes.txt >> data/test.txt

		python classification.py data/train.txt data/test.txt $classifier
	done
done