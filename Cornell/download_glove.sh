wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -d . glove.6B.zip glove.6B.100d.txt
rm -f glove.6B.zip

wget -O ./glove_s100.zip http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s100.zip
unzip -d . glove_s100.zip glove_s100.txt
rm -f glove_s100.zip
