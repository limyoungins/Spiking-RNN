NOTE:   This set of RANN models is the SECOND set. The first was a series of trial and error blunderings through getting the network working.
	Now, the network is working, but the directory and data is bloated to all hell. This is a do-over. Attempt to recreate model from [1]
	for the purpose of playing with batch/delay mismatch once the recurrent SNN layer is working (accuracy matches paper).

For all, random seed = 42397

RANN_1
	- same architecture as "old and bloated" model (previous set of RANNs) but now using some more lightning tools and seeing how far we can go
		- actually configuring settings and not just running "fit" (lightning did a lot of work for me...)
	- trained for 20 epochs using 5500 dataset
		- 10 epochs at lr=1e-3, 10 at 1e-4
	- reached 72.3% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- labeled "version_0" in lightning logs
	- post-conversion model reaches 47.0% on the test data
		- turns out, I had not been properly grabbing the previous word's spikes during recurrent inference...
		- gets worse with threshold balancing? (drops to ~30% again... but loss improves from ~17 to ~4???)

RANN_2
	- same architecture as RANN_2 but this time with different normalization
		- normalize word vectors to max value before passing to RNN, before was normalizing to sum of word vector (more info can pass through SNN?)
	- trained for 20 epochs using 5500 dataset
		- 10 epochs at lr=1e-3, 10 at 1e-4
	- reached 84.3% accuracy on the test data
	- as optimizer, uses: torch.optim.Adam
		- labeled "version_1" in lightning logs
	- post-conversion model reaches 55.0% on the test data
		- stil gets worse with threshold balancing? (drops to ~34% again... but loss improves from ~14 to ~7???)
			- number from thresh. bal. with 16 and inf. with 16 -- thresh bal. using 100 (inf. still 16) acc is 30 and loss is 2

[1] "Conversion of Artificial Recurrent Neural Networks to Spiking Neural Networks for Low-power Neuromorphic Hardware",
P. Diehl et al, 2016, https://arxiv.org/pdf/1601.04187