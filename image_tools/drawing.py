"""
image_tools/drawing.py
======================

Pure functions for drawing shapes and text onto an image.

Each function returns a copy — the original image is never modified —
so the caller can chain drawing operations safely.
"""

import cv2
import numpy as np


def draw_line(image, point1, point2, color, thickness):
    """Draw a straight line between two points."""
    return cv2.line(image.copy(), point1, point2, color, thickness)


def draw_rectangle(image, point1, point2, color, thickness):
    """Draw a rectangle. Use thickness=-1 to fill it."""
    return cv2.rectangle(image.copy(), point1, point2, color, thickness)


def draw_filled_highlight(image, point1, point2, color, opacity, mode="Blend"):
    """
    Draw a translucent colored rectangle over a region.

    Unlike draw_rectangle, this blends the color with what is underneath
    instead of replacing it, so the content stays visible.

    mode="Blend" interpolates original and annotation by opacity in [0,1].
    mode="Lesson additive" reproduces the normalized additive mask from
    Session4; it returns float [0,1] and can brighten/saturate the region.
    Other drawing functions return uint8 copies.
    """
    # Blend is true opacity; the additive choice preserves the lesson demo.
    if not 0 <= opacity <= 1:
        raise ValueError('Opacity must be between 0 and 1.')
    if mode == "Lesson additive":
        mask = np.zeros_like(image, dtype=np.float32)
        normalized_color = tuple(value / 255 for value in color)
        cv2.rectangle(mask, point1, point2, normalized_color, -1)
        return np.clip(image.astype(np.float32) / 255 + opacity * mask, 0, 1)
    if mode != "Blend":
        raise ValueError('Choose Blend or Lesson additive.')
    overlay = image.copy()
    cv2.rectangle(overlay, point1, point2, color, -1)
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)



def draw_circle(image, center, radius, color, thickness):
    """Draw a circle. Use thickness=-1 to fill it."""
    return cv2.circle(image.copy(), center, radius, color, thickness)


def draw_polyline(image, points, close_shape, color, thickness):
    """Connect a list of points. Set close_shape=True to close the outline."""
    array = np.array(points, dtype=np.int32)
    return cv2.polylines(image.copy(), [array], close_shape, color, thickness)


def draw_filled_polygon(image, points, color):
    """Fill the area enclosed by at least three points."""
    array = np.array(points, dtype=np.int32)
    return cv2.fillPoly(image.copy(), [array], color)


def draw_text(image, label, position, font_scale, color, thickness):
    """
    Draw Latin text using the built-in Hershey font.

    The built-in font does not render Persian; the UI warns about this.
    """
    return cv2.putText(
        image.copy(), label, position, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA
    )


def draw_marker(image, position, marker_kind, marker_size, color, thickness):
    """
    Draw a marker at a point.

    marker_kind must be one of the cv2.MARKER_* constants.
    """
    return cv2.drawMarker(image.copy(), position, color, marker_kind, marker_size, thickness)


def draw_arrow(image, point1, point2, color, thickness, tip_length):
    """Draw an arrow with a configurable tip length."""
    return cv2.arrowedLine(image.copy(), point1, point2, color, thickness, tipLength=tip_length)
