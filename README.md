# Frozen-Lake-DQL
Frozen Lake Environment in Processing with Deep Q-Learning <br>
Done with Processing 3.4, no libraries.

It is possible to improve considerably the speed at which the Neural Network is trained by following these two steps:
- Set global variable _nnSpace_ and comment out lines that draw the NN (line 139, class Agent; line 51, class Brain).
- Use a _while_ loop instead of an _if_ in _draw()_ method.

Colors used to draw the Neural Network are arbitrary and can be changed however in the function _render()_ (class Brain).

References: <br>
https://medium.com/@qempsil0914/zero-to-one-deep-q-learning-part1-basic-introduction-and-implementation-bb7602b55a2c
https://towardsdatascience.com/part-2-gradient-descent-and-backpropagation-bf90932c066a
https://towardsdatascience.com/part-3-implementation-in-java-7bd305faad0
