RANN_1
	- accidentally added another dimension 16 layer before the recurrent layer...
	- trained for 100 epochs using 5500 dataset (lr=1e-3)
	- reached ~68% accuracy on the test data
	- as optimizer, uses: torch.optim.SGD
RANN_2
	- removed extra layer from RANN_1 to match reference paper [1]
	- trained for ~100(?) (trained piece by piece and didn't count) epochs using 5500 dataset (lr=1e-3)
	- reached ~70% accuracy on the test data
	- as optimizer, uses: torch.optim.SGD

RANN_3
	- same architecture as RANN_2 this time with new optimizer (should have more stable performance improvement during training)
	- trained for 100 epochs using 5500 dataset (lr=1e-3)
	- reached 76.5% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam

RANN_4
	- same architecture as RANN_3 this time with cuda working so it runs on the GPU -- actually other models run fine on cuda too (was code error...)
	- trained for 110 epochs using 5500 dataset (lr=1e-3)
	- reached 75.6% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam

RANN_5
	- same architecture as RANN_4 this time with 0s along the diagonal (should have been this way from the go...)
	- trained for 150 epochs using 5500 dataset (lr=1e-4) <------ see RANN_6 (made on same day as RANN_6 -- assume I forgot to change after I experimented briefly on discarded copy of RANN_4)
	- reached 65.4% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam

RANN_6
	- same architecture as RANN_5 this time with less training epochs because RNN_5 had bad performance on the SNN (maybe overfit? -- kept guessing (almost only) 0, 1, or 4...)
	- trained for 75 epochs using 5500 dataset (lr=1e-4) <------ REALIZED DURING RANN_7 construction... should've been 1e-3
	- reached 61.2% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- still guesses (almost) only 0, 1, or 4 ?????
		- noticing that all of the others have "preferred" answers as well... not always the same, but usually 1 the most, and sometimes a 2nd/3rd preferred secondarily
		- best model gets ~30% when transferred to SNN (RANN_4), but still has above behavior

RANN_7
	- same architecture as RANN_6 but let's stop doing inference on our 0 vector -- also prepare input such that we can do batching (pad with zero vectors to eq. lengths)
		- actually... would need to rebuild forward function to accomodate batching... not doing batching yet (should not change results -- only speed -- anyway)
		- also fixing learning rate
	- trained for 20 epochs using 5500 dataset (lr=1e-3)
	- reached 71.8% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- same behavior...

RANN_8
	- same architecture as RANN_7 but now let's restrict the output projection layer (16->6) to positive weights (activations are always negative...)
	- trained for 30 epochs using 5500 dataset (lr=1e-3)
	- reached 72.5% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- same behavior...
		- preference for answering 0 persists despite variation in input/batch size (occasionally guesses 1 or 3 as well)

RANN_9
	- same architecture as RANN_8 but now let's allow biases in all layers but the one later converted to an SNN (the recurrent layer)
	- trained for 20 epochs using 5500 dataset (lr=1e-3)
	- reached 20.6% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- absolute disaster, immediately giving up -- training started going backwards...

- REALIZED HERE I WAS GRABBING THE WRONG SPIKES TO CHECK THE OUTPUT FROM THE SNN... STILL GENERALLY THE SAME BAHAVIOR AS DESCRIBED, BUT NOT AS BAD (why do I never see 2 guessed?)

RANN_10
	- same architecture as RANN_8 but now let's actually use a softmax on the output like the paper... 
		- also, why was I restricting the output weights to be positive???  (not doing that anymore)
		- sure, the output is negative, but the paper doen't mention anything like it... why is that necessarily a problem?
	- trained for 50 epochs using 5500 dataset (lr=1e-3)
	- reached 41.9% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- another disaster... with softmax on output, think I need to change the loss function (was using CrossEntropyLoss)

RANN_11
	- same architecture as RANN_10 but, after some research, I realized you need to use Negative Log-Likelihood Loss (NLLLoss) not Cross Entropy Loss (because last layer has a softmax)
		- must actually use LogSoftmax because pytorch expects the log to be performed before plugging into NLLLoss (docs said something about "mathematical stability")
		- also added shuffling to dataloader
		- also possibly fixed how target was represented
			- now a single value giving correct classification index for target instead of zero vector of length [num classifications] with a 1 in the proper index
	- trained for 50 epochs using 5500 dataset (lr=1e-3)
	- reached 74.3% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- looking good.. but oh wait... SNN STILL DOES THE SAME THING ???

RANN_12
	- same architecture as RANN_11 but now fixing extremely annoying issue where some data becomes empty after NULTIPLE EPOCHS ??? and breaks training
		- problem with shuffling ???
	- trained for 100 epochs using 5500 dataset (lr=1e-3)
	- reached 73.4% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- issue from above appears to occur when the input is not normalized and threshold balancing has not been performed
			- too many spikes in some channels -- overloading
		- still only reaching ~23% accuracy on SNN, but loss is much closer than before (~2 instead of ~50 -- trained ANN loss is ~.8)
			- still same over-focused distrubution, but as mentioned before it's better (for SNN, btw)

RANN_13
	- same architecture as RANN_12 but with word2vector model actually used in paper (not yet trained on ALL of wikipedia -- first billion(? maybe 3?) words)
	- trained for 50 epochs using 5500 dataset (lr=1e-3)
	- reached 74.7% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam

[1] "Conversion of Artificial Recurrent Neural Networks to Spiking Neural Networks for Low-power Neuromorphic Hardware",
P. Diehl et al, 2016, https://arxiv.org/pdf/1601.04187