# VALIDATION FOLDER
# MAKEFILE FOR VALIDATING TESTSETS AND FULL DATASETS BOTH SEQUENTIAL AND PARALLEL


# Creating validation files for a sequential run of 2 pixel datasets

2pixelsSeq:
	rm ../data/sahara2val.data || true
	rm ../data/peru2val.data   || true
	futhark c bfast-irreg.fut
	./bfast-irreg   < ../data/sahara2pix.in > ../data/sahara2val.data	
	./bfast-irreg   < ../data/peru2pix.in   > ../data/peru2val.data	
	rm bfast-irreg bfast-irreg.c || true


# Creating validation files for a parallel run of 2 pixel datasets

2pixelsPar:
	rm ../data/sahara2val.data || true
	rm ../data/peru2val.data   || true
	futhark c bfast-irreg.fut
	./bfast-irreg   < ../data/sahara2pix.in > ../data/sahara2val.data
	./bfast-irreg   < ../data/peru2pix.in   > ../data/peru2val.data	
	rm bfast-irreg bfast-irreg.cl || true


# Creating validation files for a sequential run of the full datasets

fullSeqSahara:
	rm ../data/saharaval.data || true
	futhark c bfast-irreg.fut
	./bfast-irreg   < ../data/sahara.in > ../data/saharaval.data	
	rm bfast-irreg || true

fullSeqPeru:
	rm ../data/peruval.data   || true
	futhark c bfast-irreg.fut
	./bfast-irreg   < ../data/peru.in > ../data/peruval.data	
	rm bfast-irreg || true


# Creating validation files for a parallel run of the full datasets

fullParSahara:
	rm ../data/saharaval.data || true
	futhark c bfast-irreg.fut
	./bfast-irreg   < ../data/sahara.in > ../data/saharaval.data	
	rm bfast-irreg || true

fullParPeru:
	rm ../data/peruval.data   || true
	futhark c bfast-irreg.fut
	./bfast-irreg   < ../data/peru.in   > ../data/peruval.data	
	rm bfast-irreg || true



