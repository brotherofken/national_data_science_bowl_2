import numpy as np
from skimage.transform import rotate, warp, AffineTransform, SimilarityTransform
from multiprocessing import Pool, cpu_count, Process, Queue
from skimage import img_as_float
import theano



def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    Copied this function from Kaggle NDSB winners: https://github.com/benanne/kaggle-ndsb
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = SimilarityTransform(translation=-center_shift)
    tform_center = SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


class Transformer(Process):

    def __init__(self,
                 images,
                 labels,
                 epoch_queue,
                 batch_size=64,
                 random_state=42,
                 mean=None):
        super(Transformer, self).__init__()
        self.images = images
        self.n_imgs = self.images.shape[0]
        self.labels = labels
        self.mean = mean
        self.queue = epoch_queue
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)

    def color_transform(self, image, sigma1=.02, sigma2=.03, sigma3=.06, sigma4=.06):
        h, w, c = image.shape
        noise1 = 0. if not sigma1 else self.rng.normal(scale=sigma1, size=c)
        noise2 = self.rng.normal(scale=sigma2, size=c)
        noise3 = self.rng.normal(scale=sigma3, size=c)
        noise4 = self.rng.normal(scale=sigma4, size=c)
        # vignette
        x = np.repeat(
            np.arange(w, dtype=theano.config.floatX)[..., np.newaxis] / w - 0.5,
            h,
            axis=1)[..., np.newaxis] * noise3.T
        y = np.repeat(
            np.arange(h, dtype=theano.config.floatX)[..., np.newaxis] / h - 0.5,
            w,
            axis=1)[..., np.newaxis] * noise4.T
        return image + noise1 + noise2 * np.cos(image * np.pi / 2) + x + y

    def _transform(self, image):
        # crop augmentation
        crop_x, crop_y = self.rng.randint(0, high=10, size=2)
        image = image[crop_x:crop_x+90, crop_y:crop_y+90, :]

        image = self.color_transform(image)
        target_shape = (90, 90)
        log_zoom_range = [np.log(z) for z in (1/1.6, 1.6)]

        stretch = np.exp(self.rng.uniform(-np.log(1.3), np.log(1.3)))
        zoom = np.exp(self.rng.uniform(*log_zoom_range))

        zoom_x = zoom / stretch
        zoom_y = zoom * stretch

        rot_degree = self.rng.randint(0, 360)
        shear = self.rng.randint(-40, 40)
        #rows, cols, c = image.shape
        #trows, tcols = target_shape
        #shift_x = cols / 2.0
        #shift_y = rows / 2.0
        shift_x, shift_y = self.rng.randint(low=-10, high=10, size=2)
        center, ucenter = build_center_uncenter_transforms(image.shape)
        form_augment = ucenter + (AffineTransform(
            scale=(1/zoom_x, 1/zoom_y),
            rotation=np.deg2rad(rot_degree),
            translation=(shift_x, shift_y),
            shear=np.deg2rad(shear)
        ) + center)

        image = warp(image, form_augment.inverse, output_shape=target_shape, mode='constant', order=1)
        # flip verticaly with 1/2 probability
        # if self.rng.randint(2):
        #     image = image[::-1, ...]
        # flip horizontaly
        if self.rng.randint(2):
            image = image[:, ::-1, ...]
        # r, g, b = self.rng.randint(2, size=3)
        # if r:
        #    image[:, :, 0] = image[:, :, 0] + self.rng.randint(-30, 30)/255.
        # if g:
        #    image[:, :, 1] = image[:, :, 1] + self.rng.randint(-30, 30)/255.
        # if b:
        #    image[:, :, 2] = image[:, :, 2] + self.rng.randint(-30, 30)/255.

        return image[np.newaxis, ...].astype(theano.config.floatX)

    # def run(self):
    #     while True:
    #         shuffle_idx = self.rng.permutation(np.arange(self.images.shape[0]))
    #         transformed = np.vstack(map(self._transform, self.images))[shuffle_idx]
    #         self.queue.put((np.rollaxis(transformed, 3, 1), self.labels[shuffle_idx]))

    def shuffle(self):
        shuffle_idx = self.rng.permutation(np.arange(self.n_imgs))
        self.images = self.images[shuffle_idx]
        self.labels = self.labels[shuffle_idx]

    def run(self):
        n_batches = int(np.ceil(self.n_imgs / self.batch_size))
        self.shuffle()
        while True:
            for idx in xrange(n_batches - 1):
                X_batch = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
                y_batch = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
                # transformed = np.vstack(map(self._transform, X_batch))
                #transformed = X_batch
                #self.queue.put((np.rollaxis(transformed, 3, 1), y_batch))
                self.queue.put((X_batch[:, np.newaxis, ...], y_batch))
            self.shuffle()
            self.queue.put((None, None))
