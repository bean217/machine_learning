import tensorflow as tf
import optimizers


from datetime import datetime


#########################################
# Multi-Layer Perceptron Implementation #
#########################################


class FlexibleDenseLayer(tf.Module):
    def __init__(self, out_dim: tf.uint16, activation=tf.nn.relu, name=None):
        super().__init__(name=name)
        # number of output dimensions
        self.out_dim = out_dim
        # the activation function
        self.activation = activation
        # boolean to check if the weights/biases for this layer have been initialized
        self.__is_built = False
    
    def __call__(self, x: tf.Tensor):
        # if the layer weights and biases have not yet been initialized
        if not self.__is_built:
            # create weights
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.out_dim]),
                name='%s/w' % self.name
            )
            # create biases
            self.b = tf.Variable(
                tf.zeros([self.out_dim]),
                name='%s/b' % self.name
            )
            # params have been initialized
            self.__is_built = True
        
        # perform activation
        return self.activation(tf.add(tf.matmul(x, self.w), self.b))


class MLP(tf.Module):
    def __init__(self, 
                 optimizer,
                 loss, 
                 architecture: list, 
                 activation=tf.sigmoid,
                 name=None):
        super().__init__(name=name)
        # optimizer
        self.optimizer = optimizer
        # loss function
        self.loss = loss
        # architecture is a list of integers with layer outputs
        self.layers = []
        for i, out_dim in enumerate(architecture):
            self.layers.append(
                FlexibleDenseLayer(out_dim=out_dim, activation=activation, name="fdl_%i" % (i+1))
            )

    def __call__(self, x: tf.Tensor):
        # propagate output
        for layer in self.layers:
            x = layer(x)
        return x
    
    @tf.function
    def train(self, batch_X, batch_y):
        # calculate loss and record gradients
        with tf.GradientTape() as t:
            loss = self.loss(self(batch_X), batch_y)
        
        # calculate the gradients
        gradients = t.gradient(loss, self.trainable_variables)
        
        # apply the gradients
        self.optimizer.apply_gradients(gradients, self.trainable_variables)
        return loss
    
    #@tf.function
    def train_loop(self, train_X, train_y, epochs, batch_size=1):
        # convert training data to a tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        # shuffle the dataset
        train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality()).batch(batch_size)

        # iterate over epochs
        for epoch in tf.range(epochs):
            # tf.print(f"Start of epoch", epoch)
            epoch_loss = 0.
            # iterate over batches
            for step, (batch_X, batch_y) in enumerate(train_dataset):
                # perform a training step
                step_loss = self.train(batch_X=batch_X, batch_y=batch_y)
                # tf.print("Training loss (for 1 batch) at step ",step,": ",step_loss,sep='')
                # tf.print("Total samples seen:",(step+1) * batch_size)


def mlp():
    train_X = tf.constant([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    train_y = tf.constant([[0.],[0.],[0.],[1.]])

    mlp = MLP(optimizer=optimizers.GradientDescent(learning_rate=1e-1), 
              loss=lambda x, y: tf.reduce_sum(tf.square(y - x)), 
              architecture=[1],
              activation=tf.sigmoid)
    
    start = datetime.now()
    mlp.train_loop(train_X, train_y, epochs=1000, batch_size=4)
    end = datetime.now()
    print()
    print("Trained in:", (end - start).total_seconds(), "seconds")
    print(mlp(train_X))


###############################################
# Convolutional Neural Network Implementation #
###############################################


def main():
    pass

if __name__ == '__main__':
    main()