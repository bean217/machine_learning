import tensorflow as tf

class GradientDescent(tf.Module):
    def __init__(self, learning_rate=1e-3):
        # Initialize optimizer with learning rate
        self.learning_rate = learning_rate
        self.title = f'Gradient Descent Optimizer: learning rate={learning_rate}'
    
    def apply_gradients(self, gradients, trainable_variables):
        # Update trainable variables with basic gradient descent
        for g, tv in zip(gradients, trainable_variables):
            tv.assign_sub(self.learning_rate * g)