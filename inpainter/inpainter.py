import matplotlib.pyplot as plt


class Inpainter():
    def __init__(self, image, mask, patch_size=9):
        self.image = image
        self.mask = mask
        self.patch_size = patch_size

    def inpaint(self):
        # TODO: main method
        plt.imshow(self.image)
        plt.show()
        return self.image
