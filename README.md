# GRU4Rec: Keras implementation
GRU4Rec is a **recommendation algorithm** specifically developed for session-based problems. Recommendations are calculated taking into account the whole history of user interactions and also the temporal order in which they have occurred.

Each session is represented as a sequence of items, or item IDs (e.g. visited pages on a website). The input to the network is the one-hot vector representing the current item and the target to predict is the next item in the sequence - i.e., the next item the user will interact with. Sample tasks of this type are next click prediction and intent prediction.

This Keras implementation is meant to provide a quick and easy-to-use experimental version. The model is quite non-standard, so more low-level frameworks like Tensorflow can be used to provide additional flexibility. The implementation is based on the original article and on the Tensorflow implementation at https://github.com/Songweiping/GRU4Rec_TensorFlow.

## Model architecture review

In general, sessions may have widly varying lengths, from 2 to several hundreds. In these cases, padding is typically used to reduce all sequences to the same fixed length. The authors of GRU4Rec, however, choose another approach, namely session-parallel mini-batches, which has the advantage of being faster. The batch creation method is explained very clearly in the original article. Essentially, each element in the batch contains a single item for one particular session, the corresponding target being the next item in the same session. At the following iteration, each batch element is updated to contain the following item in the same session. When one session finishes, the corresponding place in the batch is immediately occupied by the first element of another session, and so on. This approach has the advantage of not posing any constraints on the maximum or minimum sequence length.

A GRU layer (or multiple stacked GRU layers) is used with only **one timestep**. The sequence memory is still maintained thanks to the fact that the network is **stateful** - i.e., the hidden state is not reset when the batch changes. The hidden state has shape (batch_size, n_hidden), where batch_size is the batch_size and n_hidden is the number of hidden units. So, each session in the batch has its own dedicated hidden state of size n_hidden. When one of the sessions in the batch finishes, the corresponding hidden state is reset to 0.

The GRU layer takes as input the one-hot representations directly. An additional embedding layer can optionally be added in-between, but the authors reported it did not improve performance (at least when the number of items is very large, like hundreds of thousands, as in many real-world recommender systems).

As a loss function, categorical cross-entropy may be used, since this is technically a multi-class classification problem. However, the authors propose two different losses, **BPR** and **TOP1**, that take into account the difference in score between the positive class (item) and the negative classes and seem therefore more suitable for a recommendation framework.

In both these two losses, for each sample s, Ns other items are sampled (with Ns small, say of the order of the batch size) and the score of the ground truth item is compared against the scores of the sampled items ("negative samples") instead of being just considered on its own as in the categorical cross-entropy loss.

The sampling strategy is not specified explicitly in the article. The Tensorflow implementation takes the other items in the batch as negative samples, which has the advantage of being easier to implement and making computations more efficient.

In the BPR loss, each sample s contributes to the final loss by an amount of

![alt text](https://github.com/flowel1/gru4rec-keras/blob/master/images/bpr.svg)

where si is the ground truth for sample s (= index of the next item in the session) and the sj are the indices of the negative samples; the notation sj underlines the fact that the indices of the sampled items may vary across different samples.

In the TOP1 loss, item i's constribution is instead given by

![alt text](https://github.com/flowel1/gru4rec-keras/blob/master/images/top1.svg)

These losses are not readily available in Keras. 

## Experiments with few items
For the batch-based sampling procedure to be valid, the ground truths (next items) for the different elements in the batch would have to be all different. If the number of items is much larger than the batch size, as in many real-world cases, then we can be quite confident that this assumption holds, at least approximately (although there may be extremely popular items that are present multiple times). However, if the total number of items is of the order of the batch size, then there could be a high probability that the ground truths for different items within the same batch are the same. In these cases, one may consider avoiding the sampling and use all items besides the target item as negative samples. In this case, the following formula can be exploited to calculate the BPR and TOP1 losses:

![alt text](https://github.com/flowel1/gru4rec-keras/blob/master/images/loss-formula.svg)

where y_true_s is the one-hot encoding representation of the next item for the current sample (ground truth).








