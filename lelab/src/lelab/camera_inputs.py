"""Physical camera form metadata and native policy camera selection capabilities.

Generated empty_camera_N slots are model padding, never physical capture devices.
This module never changes the configuration passed to policy.from_pretrained.
"""
from copy import deepcopy
import re

_PLACEHOLDER = re.compile(r'(?:observation\.images\.)?empty_camera_\d+\Z')
# Verified against the installed policies' prepare_images/_preprocess_images methods.
MISSING_IMAGE_POLICIES = frozenset({'smolvla', 'pi0', 'pi05'})


def is_placeholder_camera(name):
    return bool(_PLACEHOLDER.fullmatch(name))


def physical_camera_summary(config, *, policy_ref=None):
    result = deepcopy(config)
    result['input_features'] = {
        key: value for key, value in (result.get('input_features') or {}).items()
        if not (value.get('type') == 'VISUAL' and is_placeholder_camera(key))
    }
    # These policies resize images themselves. Form metadata describes a capture
    # fallback; the UI prefers the saved physical camera's actual capture settings.
    if config.get('type') in MISSING_IMAGE_POLICIES and (
        config.get('resize_imgs_with_padding') is not None or config.get('image_resolution') is not None
    ):
        for value in result['input_features'].values():
            if value.get('type') == 'VISUAL':
                value['shape'] = [3, 480, 640]
    return result


def camera_selection_policy(config):
    return {'optional': config.get('type') in MISSING_IMAGE_POLICIES, 'min_cameras': 1}


def validate_physical_camera_selection(config, cameras):
    placeholders = [name for name in cameras if is_placeholder_camera(name)]
    if placeholders:
        raise ValueError(f'Model padding inputs cannot be bound to hardware: {placeholders}; refresh the camera form')
    if config.get('type') not in MISSING_IMAGE_POLICIES:
        return  # Other policies retain their original mandatory-camera contract.
    allowed = {
        key.rsplit('.', 1)[-1] for key, value in (config.get('input_features') or {}).items()
        if value.get('type') == 'VISUAL' and not is_placeholder_camera(key)
    }
    if not cameras:
        raise ValueError('At least one physical camera must be enabled')
    unexpected = set(cameras) - allowed
    if unexpected:
        raise ValueError(f'Unknown physical camera inputs: {sorted(unexpected)}')
    indices = [settings.get('camera_index', settings.get('index_or_path')) for settings in cameras.values()]
    known = [str(index) for index in indices if index is not None]
    if len(set(known)) != len(known):
        raise ValueError('Bind each enabled model input to a different physical camera')
