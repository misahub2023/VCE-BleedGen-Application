import cv2
import numpy as np
import logging
import tensorflow as tf


class CAM:
    '''
    Base Class
    '''
    def __init__(self, model, device='cuda', preprocess=None, layer_name=None):
        if preprocess is None:
            logging.warning('Your image preprocess is None, if your preprocess '
                            'is wrapped in your Keras model, you can ignore '
                            'this message.')

        if layer_name is None:
            raise ValueError('You should specify layer name !!!')
        else:
            try:
                layer = model.get_layer(layer_name).output
            except:
                raise ValueError(f'There is no layer named "{layer_name}" in the model')

        self.layer_name = layer_name
        self.prep = preprocess
        self.device = 'GPU' if device == 'cuda' else 'CPU'
        self.model = tf.keras.Model(model.input,
                                    {'output': model.output,
                                     'feature': layer})

    def get_heatmap(self, img):
        pass

    def _check(self, feature):
        if feature.ndim != 4 or feature.shape[2] * feature.shape[3] == 1:
            raise ValueError(f'Got invalid shape of feature map: {feature.shape}, '
                              'please specify another layer to plot heatmap.')



class EigenCAM(CAM):
    def __init__(self, model, device='cuda', preprocess=None, layer_name=None):
        super().__init__(model, device, preprocess, layer_name)

    def get_heatmap(self, img):
      with tf.device(self.device):
          output = self.model(img[None] if self.prep is None else self.prep(img)[None])
          feature = tf.transpose(output['feature'], [0, 3, 1, 2])
          self._check(feature)

          s, u, v = tf.linalg.svd(feature, full_matrices=True)
          vT = tf.transpose(v, [0, 1, 3, 2])

          cam = u[..., 0, None] @ s[..., 0, None, None] @ vT[..., 0, None, :]
          cam = tf.reduce_sum(cam, 1)

          cam -= tf.reduce_min(cam)
          cam = cam / tf.reduce_max(cam) * 255
          cam = cam.numpy().transpose(1, 2, 0).astype(np.uint8)
          cam = cv2.resize(cam, img.shape[:2][::-1])
          cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
          cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

          if img.max() <= 1.0:
              img = (img * 255).astype(np.uint8)
          else:
              img = img.astype(np.uint8)

          overlay = cv2.addWeighted(img, 0.6, cam, 0.4, 0)

      return output['output'], overlay
