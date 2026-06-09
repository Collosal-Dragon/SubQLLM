First, the text inputs are converted into tokens - it turns all of the words into a series of vectors in very high dimensions which the machine can understand.
This process is called embedding vectors.

These vectors are structured in such a way that each direction can correspond to a different feature in the text.
For example, if we do $$\text{E(King) - E(Queen)}$$ where $\text{E(Word)}$ denotes the embedded vector for a word, it will be very similar to $$\text{E(Man) - E(Woman)}\\$$

Each token's vector exists in the $\textbf{Embedding Space}$, which has a very high number of dimensions (the same number of dimensions as the word vector)

Words with similar meanings should be roughly in the same direction. For example, Cat and Kitten would be in roughly the same direction.

Now, let us look at the uses of the dot product.
The dot product can measure how similar two vectors can be.
The dot product is done by the following method:
$$
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_n\end{bmatrix} \cdot \begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ \vdots \\ z_n \end{bmatrix} = x_1z_1 + x_2z_2 + \dots + x_nz_n
$$

If both of the vectors are in the same direction, this will lead to a large positive number. If it leads to a negative number, they are in opposite directions, and if it is $0$ then the vectors are perpendicular.

For example, if we do $\text{Cat} \cdot \text{Kitten}$ this will lead to a large positive number as they are both in the same direction. 


Now, if we do something like $\text{E(Cats)} - \text{E(Cat)}$ we will get something that will be similar to a $\vec{plural}$ vector.

If we dot $\vec{plural}$ with $\vec{Octopi}$ we will get a positive number but if we dot $\vec{plural}$ with $\vec{Octopus}$ we will get a negative number.

If we keep dotting $\vec{plural}$ with the embedded vectors of increasing integers like $\text{E(1)}$ $\text{E(2)}$ $\text{E(3)}$ etc. we will get increasingly large answers.

A useful function that comes up later is the softmax function. The softmax function is defined as
$$
\sigma \text{(z) = } \frac{e^{z_i}}{\sum{e^{z_j}}}
$$
This turns random data into a probability distribution and normalises the data.
<br>
<br>
<br>



Next, let's imagine that there is a user prompt. What is given to the machine looks something like:

$$
\text{This is a conversation between a user and a helpful, very knowledgeable AI assistant.} \\
\text{User : Give me some ideas for what to do when visiting Santiago.} \\
\text{AI Assistant:} \\
$$
This text is then passed through a transformer.
A transformer is made out of many Attention Layers and MLP layers (Multilayer Perceptron layers) which are stacked one after another.
At the end of the transformer, a series of vectors is produed. While passing through the transformer, each of the vectors which corresponded to a token gathered information about the other vectors, changing itself based on the context. This part is done with the attention layer.
After the transformer is done proecssing, it multiplies the final vector by the $\textbf{unembedding matrix} \text{ } \mathbf{W_u}$, which produces a vector of raw, unnormalized scores for each possible word.
These raw inputs are called $\textbf{logits}$.

Temperature is a constant that is used to calculate the probabilities for each word.
The probabilites are calculated by the formula
$$
\text{Probability = softmax(}\frac{\text{Logits}}{T})
$$
By changing the temperature, we can change how much the function gives to the smaller values.
If T equals a small number like $0.1$ or $0.2$, a lot of the probability is given to the highest digit. This makes the machine always pick the safest option. If we turn up the temperature, the machine starts to become more random and creative because the probabilities for small words increases.

<br><br><br>
So how does the attention layer work?

First, an important piece of information is that when the vectors are embedded, they also encode the position of the word - the vectors can both tell you what the word is and where it is in the context
Let us refer to the embedded vectors as $\vec{E}$, and we will use the example $\text{a fluffy blue creature roamed the verdant forest}$.

For the first step, it is similar to a word (like creature) asking whether there are any adjectives behind it. In the example, $\text{fluffy}$ and $\text{blue}$ are "answering" the question.
The question can be represented by a Query Matrix. We will refer to it as $\mathbf{W_Q}$.
We can then do $$Q_i = W_Q \vec{E_i}$$

The matrix $\mathbf{W_Q}$ maps the very high dimensional embedding space onto lower dimensions, which we will come back to later.

There is also a second matrix called the $\textbf{Key Matrix}$ which we will denote by $\mathbf{W_K}$.
We use the same method as before where we do $$K_i = W_K \vec{E_i}$$
We can think of these Key Vectors as answers to the questions asked by the Query Vectors.
The Key Matrix maps the embedded vectors to the $\textbf{same}$ dimension as the Query Matrix

The only reason we move down dimensions is because the Embedded Vectors hold every piece of information - their semantic meanings and many other details that are unnecesarry. We map them down to lower dimensions to isolate each of the components of the word - one question might ask about its definition, while another may ask about its tone.
The reason that we move them into the $\textbf{same}$ dimension is so that we can check their similarity.
![Key and Query Vectors](LLM_notes_images\KQvectors.png)
We can use the Dot Product from before, and it is possible because they are in the same dimension.
If the final product is a large positive number, this means that the Key Vector strongly "answers" the question by the Query Vector. The large dot product shows the transformer that this specific query relates these two words.

In our example, the words "fluffy" and "blue" would strongly relate to the word "creature", so the key query dot products would be high. The correct terminology is that "fluffy" and "blue" $\textbf{attend to}$ "creature".
Similarly, the embedding of  "the" would not attend to the embedding of "creature"

Note that the total number of dot products calculated is $n^2$ where $n$ is the number of tokens inside the sentence.

Instead of each of the columns ranging from $-\infty to \infty$, we want them to act like weights, which means we want them to act like a probability distribution.
This is because we want to know the importance of each value compared to the rest. 40 might be 99% (which means that the two words are strongly related) and -12 might be 1%.

In this situation we can apply the softmax function to the values to normalise them.
Next, we replace each of the columns with the values we calculated by using softmax.

In the original paper "Attention is all you need" they only use one function
$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{K^T Q}{\sqrt{d_k}})V
$$
Here, $T$ stands for transposing K, not temperature. $K^T Q$ is a nice way to represent all of the dot products between the Key Vectors and the Query Vectors.
The $d_k$ stands for the number of dimensions. The reason why we divide bt $\sqrt{d_k}$ is to stop the softmax function from blowing up when we get to very high dimensions.
However,
