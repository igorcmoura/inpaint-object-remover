import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2grey
from skimage.filters import laplace
from scipy.ndimage.filters import convolve


class Inpainter():
    def __init__(self, image, mask, patch_size=9, plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.priority = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()

            # TODO: find max priority patch
            # TODO: find closest patch
            # TODO: copy data
            # TODO: check continue
            keep_going = False

        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = inverse_mask.reshape(height, width, 1) \
            .repeat(3, axis=2)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = white_region.reshape(height, width, 1) \
            .repeat(3, axis=2)
        image += rgb_white_region

        plt.imshow(image)
        plt.draw()
        plt.pause(0.01)  # TODO: check if this is necessary

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and the working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]

        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([height, width])

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # TODO: check if scipy's laplace filter is faster than scikit's

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        self.data = np.fabs(normal*gradient).sum(axis=2)

    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[.25, .5, .25], [0, 0, 0], [-.25, -.5, -.25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((y_normal, x_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2grey(self.working_image)

        gradient = np.array(np.gradient(grey_image))*(1 - self.working_mask)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            y_gradient = self._patch_data(gradient[0], patch)
            x_gradient = self._patch_data(gradient[1], patch)

            max_gradient[point[0], point[1], 0] = y_gradient.max()
            max_gradient[point[0], point[1], 1] = x_gradient.max()
        return max_gradient

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    @staticmethod
    def _patch_area(patch):
        return (patch[0][1]-patch[0][0]) * (patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]
