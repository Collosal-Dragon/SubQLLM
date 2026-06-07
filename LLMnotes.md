First, the text inputs are converted into tokens - it turns all of the words into a series of vectors in very high dimensions which the machine can understand.
This process is called embedding vectors.

These vectors are structured in such a way that each direction can correspond to a different feature in the text.
For example, if we do $$\text{E(King) - E(Queen)}$$ where $\text{E(Word)}$ denotes the embedded vector for a word, it will be very similar to $$\text{E(Man) - E(Woman)}\\$$

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



Next, let's imagine that there is a user prompt. What is given to the machine looks something like:

$$
\text{This is a conversation between a user and a helpful, very knowledgeable AI assistant.} \\
\text{User : Give me some ideas for what to do when visiting Santiago.} \\
\text{AI Assistant:} \\
$$
This text is then passed through a transformer.
A transformer is made out of many Attention Layers and MLP layers (Multilayer Perceptron layers) which are stacked one after another.
At the end of the transformer, a series of vectors is produed. While passing through the transformer, each of the vectors which corresponded to a token gathered information about the other vectors, changing itself based on the context. This part is done with the attention layer.

So how does the attention layer work?

