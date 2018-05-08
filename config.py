

class Config:

	STOP_WORDS = ['a', '.', 'the', 'with', 'of', 'in', 'is', 'on']

	ENCODER_PATH = 'models/ewc-encoder.pkl'
	DECODER_PATH = 'models/ewc-decoder.pkl'

	VOCAB_PATH = 'models/vocab.pkl'

	EMBED_SIZE = 256
	HIDDEN_SIZE = 512
	NUM_LAYERS = 1