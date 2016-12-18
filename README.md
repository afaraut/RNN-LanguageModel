# RNN-LanguageModel

###		Project : Building a Language Model Using Recurrent Neural Networks 
###		Author : Anthony FARAUT

# Files : 
	- generate_data.py : Re-generate the word2vec file in order to reduce its size.
	- nrr.py : Recurrent Neural Networks.

# Data :
	- The corpus files
	- The word2vec files (gloves)

# Example of use : 

	python nrr.py 
	Using gpu device 0: Tesla K80 (CNMeM is disabled, cuDNN 5004)
	Using Theano backend.
	Epoch 1/10
	# ---------------------
	Perplexity on epoch 1 -> 1.37084873552
	# --------------------- End of the epoch
	202s - loss: 0.9291
	Epoch 2/10
	# ---------------------
	Perplexity on epoch 2 -> 1.29489823736
	# --------------------- End of the epoch
	202s - loss: 0.4877
	Epoch 3/10
	# ---------------------
	Perplexity on epoch 3 -> 1.31177034897
	# --------------------- End of the epoch
	199s - loss: 0.4660
	Epoch 4/10
	# ---------------------
	Perplexity on epoch 4 -> 1.30676592971
	# --------------------- End of the epoch
	184s - loss: 0.4506
	Epoch 5/10
	# ---------------------
	Perplexity on epoch 5 -> 1.29016040029
	# --------------------- End of the epoch
	192s - loss: 0.4372
	Epoch 6/10
	# ---------------------
	Perplexity on epoch 6 -> 1.26558529141
	# --------------------- End of the epoch
	197s - loss: 0.4256
	Epoch 7/10
	# ---------------------
	Perplexity on epoch 7 -> 1.28655986043
	# --------------------- End of the epoch
	196s - loss: 0.4142
	Epoch 8/10
	# ---------------------
	Perplexity on epoch 8 -> 1.25569127081
	# --------------------- End of the epoch
	202s - loss: 0.4028
	Epoch 9/10
	# ---------------------
	Perplexity on epoch 9 -> 1.26269284349
	# --------------------- End of the epoch
	204s - loss: 0.3929
	Epoch 10/10
	# ---------------------
	Perplexity on epoch 10 -> 1.26207161413
	# --------------------- End of the epoch
	187s - loss: 0.3820
	Testing...
	('Test accuracy loss: ', 0.59729876170239138)
	('Train accuracy loss: ', 0.36615620031151719)
	# --- predict
	('result ', '128321.0 / 141208')
	('percentage ', 90.87374652994164)