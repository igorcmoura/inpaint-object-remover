import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import laplace


class Inpainter():
    def __init__(self, image, mask, patch_size=9, plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.confidence = None
        self.front = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_confidence()

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priorities()

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

    def _initialize_confidence(self):
        """ Initialize the target region with 0 and source region with 1

        The confidence is initially the inverse of the mask.
        """
        self.confidence = (1 - self.mask).astype(float)

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')

    def _update_priorities(self):
        self._update_confidence()
        # TODO: update data
        # TODO: calc priority

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_area = (patch[0][1]-patch[0][0]) * (patch[1][1]-patch[1][0])

            new_confidence[point[0], point[1]] = sum(sum(
                self.confidence[
                    patch[0][0]:patch[0][1]+1,
                    patch[1][0]:patch[1][1]+1
                ]
            ))/patch_area

        self.confidence = new_confidence

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
