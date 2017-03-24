"""Utility functions for Self Driving car project."""


class Utils(object):
    """Helping functions."""

    def cut_rows(self, image, up=0, down=0):
        """Remove specific rows from up and down from given image dataset."""
        return image[0 + up:160 - down, :, :]


_inst = Utils()
cut_rows = _inst.cut_rows
