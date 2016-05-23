
base:
	# creating base
	wget http://www.inf.ufpr.br/lesoliveira/padroes/simpsons.zip
	unzip simpsons.zip -d data
	rm simpsons.zip
	rm -rf data/__MACOSX/
	rm data/Train/.DS_Store
	rm data/Valid/.DS_Store
	