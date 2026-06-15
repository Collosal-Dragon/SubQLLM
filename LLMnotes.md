First, the text inputs are converted into tokens - it turns all of the words into a series of vectors in very high dimensions which the machine can understand.
This process is called embedding.

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

A useful function that comes up later is the softmax function. The softmax function is defined as
$$
\sigma \text{(z)}_i = \frac{e^{z_i}}{\sum{e^{z_j}}}
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
At the end of the transformer, a series of vectors is produced. While passing through the transformer, each of the vectors which corresponded to a token gathered information about the other vectors, changing itself based on the context. This part is done with the attention layer.
After the transformer is done processing, it multiplies the final vector by the $\textbf{unembedding matrix} \text{ } \mathbf{W_u}$, which produces a vector of raw, unnormalized scores for each possible word.
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

The only reason we move down dimensions with both of the matrices is because the Embedded Vectors hold every piece of information - their semantic meanings and many other details that are unnecessary. We map them down to lower dimensions to isolate each of the components of the word - one question might ask about its definition, while another may ask about its tone.
The reason that we move them into the $\textbf{same}$ dimension is so that we can check their similarity.
![Key and Query Vectors](LLM_notes_images\KQvectors.png)
We can use the Dot Product from before, and it is possible because they are in the same dimension.
If the final product is a large positive number, this means that the Key Vector strongly "answers" the question by the Query Vector. The large dot product shows the transformer that this specific query relates these two words.

In our example, the words "fluffy" and "blue" would strongly relate to the word "creature", so the key query dot products would be high. The correct terminology is that "creature" $\textbf{attends to}$ "fluffy" and "blue".
Similarly, the embedding of  "the" would not attend to the embedding of "creature"

Note that the total number of dot products calculated is $n^2$ where $n$ is the number of tokens inside the sentence.

Instead of each of the columns ranging from $-\infty to \infty$, we want them to act like weights, which means we want them to act like a probability distribution.
This is because we want to know the importance of each value compared to the rest. 40 might be 99% (which means that the two words are strongly related) and -12 might be 1%.

In this situation we can apply the softmax function to the values to normalise them.
Next, we replace each of the columns with the values we calculated by using softmax.

In the original paper "Attention is all you need" they only use one function
$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
Here, $T$ stands for transposing K, not temperature. $QK^T$ is a nice way to represent all of the dot products between the Key Vectors and the Query Vectors.
The $d_k$ stands for the number of dimensions of the Key and Query matrices. The reason why we divide by $\sqrt{d_k}$ is to stop the softmax function from blowing up when we get to very high dimensions.
<br>
Now, since LLMs only produces one word at a time, it cannot allow future words to affect the past words. "creature" should not affect the attention of "fluffy" and "blue". We can motivate this by imagining reading a book. What is on page 100 should not affect what is on page 50, but page 50 can definitely affect what is on page 100.
If we take a look at our grid which at this stage should look something like this:
![Current Attention table](LLM_notes_images\Attention.png)

Now, for every word that gives attends to the words behind it, we need to make them equal $0$. However, if we just set them to $0$ after we apply the softmax function, our distribution will be broken. The fix for this is to set those values to $-\infty$ before we apply the softmax function. This causes all of those values to be $0$ after we apply softmax.
This process is called $\textbf{Masking}$, and it is very useful in training.

Now, we need to actually change the embeddings. In our example, we want "fluffy" to change the embedding of "creature", possibly to make the "creature" vector positive in the fluffy dimension.

The most straightforward way to do this is to use a third matrix called the $\textbf{Value Matrix}$, which we will denote as $W_V$. We then multiply $W_V$ by the embedded vector, which results in a new vector called the $\textbf{Value Vector}$ (it will be denoted as $\vec{V}$).

So what does the value matrix actually do? The value matrix produces value vectors that extract the core concepts. We do this to shorten and summarise some of the concepts, so only the key information remains.
For example, two directions in the embedded vectors may be humans and people, but with value vectors we can compress that into one as their meaning is very similar.
<br>

Now, for each of the columns, we multiply each of the value vectors by the value we got from applying the softmax function onto the dot products, then we add all of the new value matrices together. Let us call this new vector $\Delta \vec{E_i} $. To apply the changes to the original embedded vector, we do 
$$
\vec{E_i} + \Delta \vec{E_i}
$$
which results in our changed vector for this layer of the attention head.

Now, we can apply this to each embedded vector to get a changed series of vectors.

Now, how many dimensions does the Value Matrix have? We could make it a square matrix, but this would increase the parameters by a very significant number. Instead, the most efficient way is to let the dimensions of the Value Matrix $=$ (Dimensions of Query Matrix) + (Dimensions of Key Matrix).

What we have discussed so far is called $\textbf{Self Attention}$. What is used in LLMs like GPT-3 is something called $\textbf{Multi-Head Attention}$. This is just the same as running many self attention heads in parallel. For GPT-3, it uses 96 self attention heads in each block. For each self attention head, each one has unique Query, Key, and Value matrices, which means that each head produces a unique $\Delta \vec{E^i_j}$ that we add to each embedded vector.
While this is one way that we can do it, there is a more efficient and popular way to do it.

We can split the value matrix into two different matrices, which we can call the $\textbf{Value Up Matrices} \text{ (} V_\uparrow )$ and $\textbf{Value Down Matrices} \text{ (} V_\downarrow )$.

So what do the new value matrices do?$V_\downarrow$ takes the high dimensional $\vec{E_i}$ and maps it down into the much smaller dimension $d_k$. This is where the main process happens, it compresses all of the information.
$V_\uparrow$ scales the new vector back to the size of the embedded vector so that we can add them together again.

<br>
Let us now take a deeper look into Multi-Head Attention.
We can say that we are running $h$ attention heads in parallel. Each head produces its own change vector $\Delta\vec{E_i}$ in $d_k$ dimensions, so in total we have $h$ change vectors.
What we do next is we concatenate all of the $\Delta\vec{E_i}$. Concatenating vectors looks like this:
$$
\text{Concat}(\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}) = \begin{bmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{bmatrix}
$$
Finally, we multiply the output matrix $W_O$ by the concatenated $\Delta\vec{E}$, and we add that to the original embedded vector $\vec{E}$ to get the new changed embedded vector.

Each head can learn to attend to different relationships between words - one may track which words are adjectives while another may track which words are nouns, giving the model more understanding of the words.
<br>
So why do we even need $W_O$? Why do we concatenate?
<br>

Lets focus on the concatenation first. If we didn't concatenate and instead added up all of the $\Delta \vec{E_i}$ we would have no way of communicating between the different dimensions. Dimension 1 of Head 1 would only interact with Dimension 1 of the other heads, so it cannot interact with another dimension. However, when we concatenate this enables each dimension to be able to interact with every other dimension. 
<br>
What about $W_O$?

$W_O$ is what performs each interaction between different dimensions, mixing all of the information from each head. $W_O$ also ensures that the dimensions of $\Delta\vec{E}$ equal the dimensions of $\vec{E}$. Conventionally, $d_k = \frac{d_{model}}{h}$ which means that $\Delta\vec{E}$ already has $d_{model}$ dimensions, making $W_O$ a square matrix.
<br>
Now that we are done with Attention heads, we will move on to MLPs (Multi Layer Perceptrons)
<br>
MLPs usually consist of three layers.
The first layer is a linear layer. This is a simple linear layer. This is usually
$$
W_1 \vec{E_i} + \vec{b_1}
$$
Here, $W_1$ stand for the weight matrix, $\vec{E_i}$ stand for our embedded vector and $b_1$ stands for our bias vector. $W_1$ expands the dimensions of the embedded vector from $d_{model}$ to $4\cdot d_{model}$

Next, we also have to add a non-linear function. The reason we do this is so we can deal with functions that are non linear - as an example, if we only had linear functions our MLP could not learn a function like XOR. With the non-linear layer we can learn such functions. Without non-linear functions, we could just express all of the linear functions as a singular linear function.
A simple non-linear function to use is ReLU which is 
$$
\text{ReLU}(x) = \max(0,x)
$$
However, a much better function for LLMs is the GELU function. This function uses the normal distribution, and the GELU function is
$$
\text{GELU}(x) = x\cdot P(X\leq x), \text{ } X \sim N(0,1)
$$
Another way to express the function is
$$
\text{GELU}(x) = x \cdot \frac{1}{2}(1+\text{erf}(\frac{x}{\sqrt{2}}))
$$
Here,
$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_{0}^{x}{e^{-t^2}dt}
$$
So why is GELU better than ReLU?
ReLU can sometimes cause "dead neurons", because it always causes negative values to equal 0, so sometimes it can cause some neurons to stop updating and "die". 
GELU fixes this issue as it does not fully cause negative numbers to go to zero. Instead, small negative values are usually allowed to go through while large negative values go to $0$. However, dead neurons still occur but at a much smaller rate.
GELU is also a smooth continuous function, so it handles backpropagation a little bit better.

Finally, the last layer is another linear layer,
$$
W_2 \vec{h_i} + \vec{b_2}
$$
In this step, $\vec{h_i}$ is the vector that comes out of the non-linear layer. $W_2$ scales it back down to $d_{model}$

Now, we are finally done with how transformers work. We just stack many attention layers and MLP layers together to produce a final output.

But how do we train the LLM? How do we change each and every matrix/vector to produce words?

First, lets look at the loss function.
The loss function outputs a single number that tells the model how wrong it is. It usually outputs a score where the lower it is the better it is. 
A simple type of loss function is called Cross-Entropy. It is simply
$$
-\ln(\hat{p})
$$
It does not have to be base $e$, but later this will greatly simplify a lot of calculations if we were to use $\ln$.
Note that $\hat{p}$ is always between $0$ and $1$, so $-\ln(\hat{p}) > 0$.
<br>
The next topic, backpropagation, is the hardest one and it is the key as to how the LLM actually learns.
What is backpropagation?
For an example, if the next word was supposed to be "refrigerator" but the model outputs "cat", backpropagation would lower the probability of getting "cat" and increase the probability of getting "refrigerator" in this scenario. This is essentially what backpropagation does, and it is what changes all of the weights in the machine.

Now, we have a loss $L$ and up to billions of weights. How do we know which ones to tweak to make our result more accurate?

We need to know how much each weight changes the final result - we are basically asking for 
$$
\frac{\partial L}{\partial w}
$$

The key insight is that the entire machine is just a lot of functions. It might look something like
$$
L = f_n(f_{n-1}(f_{n-2}(\cdots(f_2(f_1(x))\cdots))))
$$

The reason we can compute $\frac{\partial L}{\partial w}$ is because each of the functions only relies on the output of the inner function, so we can use chain rule again and again.

The chain rule is simply 
if $L = f(g(x))$ then
$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dx}$

Let's take a look at our model. We can say the entire model boils down to :
$$
L = -\ln(\text{softmax}(W_u\cdot(\text{Attention}(\text{MLP(Attention}(\cdots)_{\text{Last Token}})
$$
Each MLP block is 
$$
\text{MLP(}\vec{x}) = W_2(GELU(W_1\vec{x} + \vec{b_1})) + \vec{b_2}
$$
And again, each attention block is 
$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
So we have to find the derivative of each function. Let's start with the $\text{MLP}()$

So we know 
$$
\text{MLP}(\vec{x}) = \text{LF}_1(\text{GELU}(\text{LF}_2))
$$
where $\text{LF}$ stands for our linear functions.
Again, we can find the derivative for this using chain rule, and we get
$$
\text{MLP}'(x) = W_2 \cdot \text{GELU}'(W_1 \vec{x} + \vec{b_1}) \cdot W_1
$$
Now, we need to evaluate what $\text{GELU}'(\vec{x})$ is.
We know that 
$$
\text{GELU}(x) = x \cdot \frac{1}{2}(1+\text{erf}(\frac{x}{\sqrt{2}}))
$$
and 
$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_{0}^{x}{e^{-t^2}dt}
$$
To find $\text{GELU}'(\vec{x})$ we can use product rule, and we get
$$
\frac{1}{2}(1+\text{erf}(\frac{x}{\sqrt{2}})) + x\cdot\frac{1}{2}\cdot\text{erf}'(\frac{x}{\sqrt{2}})\cdot \frac{1}{\sqrt{2}}
$$
We can find $\text{erf}'(x)$ by using the Fundamental Theorem of Calculus. The FTC states that 
$$
\frac{d}{dx} \int_{a}^{x}f(t)dt = f(x)
$$
So, $\text{erf}'(x)$ equals
$$
\frac{2}{\sqrt{\pi}} \frac{d}{dx} \int_{0}^{x}e^{-t^2}dt
$$
and using the FTC this becomes
$$
\text{erf}'(x) = \frac{2}{\sqrt{\pi}}e^{-x^2}
$$
Now, putting this all together:
$$
\text{GELU}'(\vec{x}) =  \frac{1}{2}(1+\text{erf}(\frac{x}{\sqrt{2}})) + \frac{x}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}
$$
Finally,
$$
\text{MLP}'(\vec{x}) = W_2\cdot (\frac{1}{2}(1+\text{erf}(\frac{x}{\sqrt{2}})) + \frac{x}{\sqrt{2\pi}}e^{-\frac{x^2}{2}})\cdot W_1
$$
