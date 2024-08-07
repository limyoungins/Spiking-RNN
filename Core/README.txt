Codebase to examine the effect of temporal batch size/delay mismatch in recurrent spiking neural networks. Original code used to define spiking network/neurons. 
Pytorch/lightning used to train ANN. Model is reproduction of P. Diehl et al paper [1] which perform question classification. Question data and word2vec converter 
are same as used in paper. word2vec uses same model, but was trained on 2019 wikipedia (paper is from 2016).

Recurrent ANN Trainer.ipynb (jupyter file) trains ANN using question dataset and word2vec dictionary (must produce separately -- too large to upload) model is saved
in Recurrent ANN Models folder. See README in folder for details on each model. Recurrent SNN Batch Size Test.ipynb converts the ANN's recurrent layer to spiking neurons
using code from SpikingNetwork.py, which implements temporally batched computations.

Once converted model reaches about 5% conversion loss (accuracy on test set), batch size/delay mismatch will be investigated. Subsequent work should investigate LIF neurons
trained using some kind of STDP backpropagation (no longer spike-time agnostic), and perhaps learned delays.

Background: 
Optical system would benefit greatly from the ability to temporally batch calculations, but real neurons have delays as short as 1ms (especially in the visual cortex).
How do you handle calculating in temporal batches with delays shorter than the batch size? (The recurrent inputs are not available mid-batch.)

Author: Liam Young

[1] "Conversion of Artificial Recurrent Neural Networks to Spiking Neural Networks for Low-power Neuromorphic Hardware",
P. Diehl et al, 2016, https://arxiv.org/pdf/1601.04187