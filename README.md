# GRU4Rec: Keras implementation
GRU4Rec is a **recommendation algorithm** developed to deal with session-based problems: recommendations are calculated taking into account the whole history of user interactions and also the temporal order in which they have occurred.

Each session is represented as a sequence of items (e.g. visited pages on a website). The input to the network is the one-hot representation of the current item and the target to predict is the next item in the sequence - i.e., the next item the user will interact with.

This Keras implementation is meant to provide an easy-to-use experimental version. The model is quite non-standard, so more low-level frameworks can be used to provide additional flexibility.

The implementation is based on the original article and on the Tensorflow implementation at https://github.com/Songweiping/GRU4Rec_TensorFlow.

## Model architecture review

In general, sessions may have widly varying lengths, from 2 to several hundreds. To make them all processable by the network, we could use padding to reduce all sequences to the same fixed length, but the authors choose another approach, namely session-parallel mini-batches. The batch creation method is explained very clearly in the original article.

So, each element in the batch contains a session element for one particular session. When one session finishes, the corresponding place in the batch is immediately occupied by the first element of another session.
This approach has the advantage of not posing any constraints on the maximum sequence length.

A GRU layer is used with only **one timestep**. The sequence memory is still maintained thanks to the fact that the network is made **stateful** - i.e., the hidden state is maintained when the batch changes. The hidden state has shape (batch_size, n_hidden). So each session in the batch has its own dedicated hidden state of size n_hidden. When one of the sessions in the batch finishes, the corresponding hidden state is reset to 0.

As a loss function, categorical cross-entropy may be used, since this is technically a multi-class classification problem. However, the authors propose two different losses, **BPR** and **TOP1**. These losses are not readily available in Keras.

In both these two losses, for each sample s, Ns other items are sampled (with Ns small, say of the order of the batch size) and the score of the ground truth item is compared against the scores of the sampled items ("negative samples") instead of being just considered on its own as in the categorical cross-entropy loss. 

The sampling strategy is not specified explicitly in the article. The Tensorflow implementation takes the other items in the batch as negative samples.

In the BPR loss, each sample s contributes to the final loss by an amount of

- \frac{1}{N_s} \sum_{j=1}^{N_s} \log (\sigma(\hat{r}_i^{(s)} - \hat{r}_j^{(s)})))

where si is the ground truth for sample s (= index of the next item in the session) and the sj are the indices of the negative examples; the notation sj underlines the fact that the indices of the sampled items may vary across different samples.

In the TOP1 loss, item i's constribution is instead given by

\frac{1}{N_s} \sum_{j=1}^{N_s} \left\{ \sigma(\hat{r}_{s_j} - \hat{r}_{s_i}) + \sigma(\hat{r}_{s_j}^2)\right\}

To calculate the loss contribution of a given sample, we can exploit the following formula (assuming for the moment that we do no sampling, i.e., that all items are considered):

\begin{bmatrix}
\hat{r}_{s_i} - \hat{r}_1 \phantom{-1} \\ 
\hat{r}_{s_i} - \hat{r}_2 \phantom{-1} \\ 
\vdots 
\\ 
\hat{r}_{s_i} - \hat{r}_{\text{n\_items}}\\ 
\end{bmatrix} = \left\{ \begin{bmatrix}
1\\ 
1 \\ 
\vdots 
\\ 
1\\ 
\end{bmatrix} \cdot \text{y\_true}_s - I \right\} \cdot \begin{bmatrix}
\hat{r}_1 \phantom{-1} \\ 
\hat{r}_2 \phantom{-1} \\ 
\vdots 
\\ 
\hat{r}_{\text{n\_items}}\\ 
\end{bmatrix} 

where y_true_s is the one-hot encoding representation of the next item for the current sample (ground truth).








