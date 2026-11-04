"""JPEG observation color decode (vendored from XPolicyLab utils.process_data).

Only the decode path used by the websocket policy server is kept here.
Encoding helpers and unrelated data-conversion utilities are omitted.
"""

from __future__ import annotations

import numpy as np


def _cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Decoding JPEG observation colors requires opencv-python-headless "
            "(or opencv-python). Install it in the policy environment, or send "
            "already-decoded uint8 RGB arrays."
        ) from exc
    return cv2

# Two image bit stream formats exist, and both decode to RGB.
#
#   legacy    A JPEG written by handing an RGB array straight to cv2.imencode,
#             which reads its input as BGR. The stored bytes are therefore
#             channel-reversed with respect to the JPEG standard.
#   standard  A conforming RGB JPEG stamped with a COM segment holding
#             _RGB_MARKER_PAYLOAD. cv2.imdecode returns BGR for it, so decoding
#             needs exactly one swap.
_RGB_MARKER_PAYLOAD = b"XPL-RGB1"

_JPEG_PAD = 0xFF
_JPEG_STUFF = 0x00
_JPEG_TEM = 0x01
_JPEG_RST_FIRST = 0xD0
_JPEG_RST_LAST = 0xD7
_JPEG_SOI = 0xD8
_JPEG_EOI = 0xD9
_JPEG_SOS = 0xDA
_JPEG_COM = 0xFE

_COM_SEGMENT_LENGTH_BYTES = 2
_RGB_MARKER_SEGMENT_LENGTH = len(_RGB_MARKER_PAYLOAD) + _COM_SEGMENT_LENGTH_BYTES

OBS_IMAGE_KEYS = ("color", "colors", "rgb", "image")
_DECODABLE_TYPES = (bytes, bytearray, memoryview, np.ndarray, list, tuple)


def _normalize_image_buffer(image_bit):
    if isinstance(image_bit, np.ndarray) and image_bit.dtype.kind in {"S", "U"}:
        image_bit = image_bit.item() if image_bit.ndim == 0 else image_bit.tobytes()

    if isinstance(image_bit, str):
        image_bit = image_bit.encode("utf-8")
    elif isinstance(image_bit, memoryview):
        image_bit = image_bit.tobytes()

    if isinstance(image_bit, (bytes, bytearray)):
        image_bit = image_bit.rstrip(b"\0")
    elif isinstance(image_bit, np.ndarray):
        image_bit = np.ascontiguousarray(image_bit)

    return image_bit


def _byte_view(image_bit) -> memoryview:
    if isinstance(image_bit, (bytes, bytearray, memoryview)):
        return memoryview(image_bit).cast("B")

    array = np.ascontiguousarray(image_bit)
    if array.dtype != np.uint8:
        return memoryview(array.tobytes())
    return memoryview(array.reshape(-1))


def _has_rgb_marker(image_bit) -> bool:
    view = _byte_view(_normalize_image_buffer(image_bit))
    size = len(view)

    if size < 4 or view[0] != _JPEG_PAD or view[1] != _JPEG_SOI:
        return False

    offset = 2
    while offset + 4 <= size:
        if view[offset] != _JPEG_PAD:
            return False

        marker = view[offset + 1]
        if marker == _JPEG_PAD:
            offset += 1
            continue
        if marker == _JPEG_STUFF:
            return False
        if marker in (_JPEG_TEM, _JPEG_SOI, _JPEG_EOI) or (
            _JPEG_RST_FIRST <= marker <= _JPEG_RST_LAST
        ):
            offset += 2
            continue
        if marker == _JPEG_SOS:
            return False

        length = (view[offset + 2] << 8) | view[offset + 3]
        if length < _COM_SEGMENT_LENGTH_BYTES:
            return False

        if marker == _JPEG_COM and length == _RGB_MARKER_SEGMENT_LENGTH:
            payload = view[offset + 4 : offset + 2 + length]
            if bytes(payload) == _RGB_MARKER_PAYLOAD:
                return True

        offset += 2 + length

    return False


def _decode_single_image_bit(image_bit):
    cv2 = _cv2()
    image_bit = _normalize_image_buffer(image_bit)
    image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Failed to decode image bits (type={type(image_bit).__name__}, "
            f"size={getattr(image_bit, 'size', len(image_bit) if hasattr(image_bit, '__len__') else '?')})."
        )
    if _has_rgb_marker(image_bit):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _decode_image_bit_sequence(image_bits):
    frames = []
    for index, image_bit in enumerate(image_bits):
        try:
            frames.append(decode_image_bit(image_bit))
        except ValueError as exc:
            raise ValueError(f"Frame {index}: {exc}") from exc
    if not frames:
        return np.zeros((0,), dtype=np.uint8)
    return np.stack(frames, axis=0)


def decode_image_bit(image_bits):
    """Decode encoded image bit stream(s) into uint8 RGB image array(s)."""
    if isinstance(image_bits, (bytes, bytearray, memoryview, str)):
        return _decode_single_image_bit(image_bits)

    if isinstance(image_bits, np.ndarray):
        if image_bits.dtype.kind in {"S", "U", "O"}:
            if image_bits.ndim == 0:
                return _decode_single_image_bit(image_bits.item())
            return _decode_image_bit_sequence(image_bits)
        if image_bits.dtype == np.uint8:
            if image_bits.ndim == 1:
                return _decode_single_image_bit(image_bits)
            if image_bits.ndim == 2:
                return _decode_image_bit_sequence(image_bits)
        return image_bits

    if isinstance(image_bits, (list, tuple)):
        return _decode_image_bit_sequence(image_bits)

    return _decode_single_image_bit(image_bits)


def _decode_obs_image(value, camera_name, image_key):
    if value is None:
        return value
    try:
        return decode_image_bit(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Failed to decode obs['vision']['{camera_name}']['{image_key}']: {exc}"
        ) from exc


def decode_obs_images(obs):
    """Decode encoded color streams of a runtime observation, in place."""
    if isinstance(obs, (list, tuple)):
        for single_obs in obs:
            decode_obs_images(single_obs)
        return obs

    if not isinstance(obs, dict):
        return obs

    vision = obs.get("vision")
    if not isinstance(vision, dict):
        return obs

    for camera_name, camera in vision.items():
        if isinstance(camera, dict):
            for image_key in OBS_IMAGE_KEYS:
                if image_key in camera:
                    camera[image_key] = _decode_obs_image(
                        camera[image_key], camera_name, image_key
                    )
        elif isinstance(camera, _DECODABLE_TYPES):
            vision[camera_name] = _decode_obs_image(camera, camera_name, "color")

    return obs
