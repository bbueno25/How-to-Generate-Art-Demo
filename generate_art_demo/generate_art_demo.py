"""
DOCSTRING
"""
import keras.applications.vgg16
import keras.backend
import numpy
import PIL
import scipy.optimize
import time

class Evaluator:
    """
    DOCSTRING
    """
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def eval_loss_and_grads(self, x):
        """
        DOCSTRING
        """
        x = x.reshape((1, self.height, self.width, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    def grads(self, x):
        """
        DOCSTRING
        """
        assert self.loss_value is not None
        grad_values = numpy.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def loss(self, x):
        """
        DOCSTRING
        """
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

class GenerateArt:
    """
    DOCSTRING
    """
    def __init__(self):
        self.height = 512
        self.width = 512

    def __call__(self):
        content_image_path = 'images/hugo.jpg'
        content_image = PIL.Image.open(content_image_path)
        content_image = content_image.resize((self.height, self.width))
        style_image_path = 'images/wave.jpg'
        style_image = PIL.Image.open(style_image_path)
        style_image = style_image.resize((self.height, self.width))
        content_array = numpy.asarray(content_image, dtype='float32')
        content_array = numpy.expand_dims(content_array, axis=0)
        print(content_array.shape)
        style_array = numpy.asarray(style_image, dtype='float32')
        style_array = numpy.expand_dims(style_array, axis=0)
        print(style_array.shape)
        content_array[:, :, :, 0] -= 103.939
        content_array[:, :, :, 1] -= 116.779
        content_array[:, :, :, 2] -= 123.68
        content_array = content_array[:, :, :, ::-1]
        style_array[:, :, :, 0] -= 103.939
        style_array[:, :, :, 1] -= 116.779
        style_array[:, :, :, 2] -= 123.68
        style_array = style_array[:, :, :, ::-1]
        content_image = keras.backend.variable(content_array)
        style_image = keras.backend.variable(style_array)
        combination_image = keras.backend.placeholder((1, self.height, self.width, 3))
        input_tensor = keras.backend.concatenate(
            [content_image, style_image, combination_image], axis=0)
        model = keras.applications.vgg16.VGG16(
            input_tensor=input_tensor, weights='imagenet', include_top=False)                    
        layers = dict([(layer.name, layer.output) for layer in model.layers])
        content_weight = 0.025
        style_weight = 5.0
        total_variation_weight = 1.0
        loss = keras.backend.variable(0.0)
        layer_features = layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        content_loss = keras.backend.sum(keras.backend.square(
            combination_features - content_image_features))
        loss.assign_add(content_weight * content_loss)
        feature_layers = [
            'block1_conv2', 
            'block2_conv2',
            'block3_conv3', 
            'block4_conv3',
            'block5_conv3']
        for layer_name in feature_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.style_loss(style_features, combination_features)
            loss.assign_add((style_weight / len(feature_layers)) * sl)
        loss.assign_add(total_variation_weight * self.total_variation_loss(combination_image))
        grads = keras.backend.gradients(loss, combination_image)
        outputs = [loss]
        outputs += grads
        f_outputs = keras.backend.function([combination_image], outputs)
        evaluator = Evaluator()
        x = numpy.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.
        iterations = 10
        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = scipy.optimize.fmin_l_bfgs_b(
                evaluator.loss, 
                x.flatten(), 
                fprime=evaluator.grads, 
                maxfun=20
                )
            print('Current loss value:', min_val)
            end_time = time.time()
            print('Iteration %d completed in %ds' % (i, end_time - start_time))
        x = x.reshape((self.height, self.width, 3))
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = numpy.clip(x, 0, 255).astype('uint8')
        PIL.Image.fromarray(x)

    def gram_matrix(self, x):
        """
        DOCSTRING
        """
        features = keras.backend.batch_flatten(keras.backend.permute_dimensions(x, (2, 0, 1)))
        gram = keras.backend.dot(features, keras.backend.transpose(features))
        return gram     

    def style_loss(self, style, combination):
        """
        DOCSTRING
        """
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.height * self.width
        return keras.backend.sum(
            keras.backend.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def total_variation_loss(self, x):
        """
        DOCSTRING
        """
        a = keras.backend.square(x[:, :self.height-1, :self.width-1, :] - x[:, 1:, :self.width-1, :])
        b = keras.backend.square(x[:, :self.height-1, :self.width-1, :] - x[:, :self.height-1, 1:, :])
        return keras.backend.sum(keras.backend.pow(a + b, 1.25))

if __name__ == '__main__':
    generate_art = GenerateArt()
    generate_art()
