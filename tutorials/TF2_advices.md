## Tensorflow 2.0 advices

For a while, I used Tensorflow 1.4 and Keras 2.3.0. First one for its performance, second for its conveniance.
When I heard that Tensorflow 2.0 was going to encapsulate Keras better, I though I find the best of the two world: conveniance and performance.

But without the following advices, Tensorflow 2.0 with Keras style is so slow !


### 1- Forget about fit and train_on_batch

To execute a gradient descent on your model, it is usual to code with the Keras style:  

```
model.fit(x_batch, y_batch, epochs=..) # Training multiple times on the batch
```
Or
```
model.train_on_batch(x_batch, y_batch) # Training once
```

But this is so slow. To speed-up, it is recommanded to write your own gradient descent method and use **@tf.function decorator**