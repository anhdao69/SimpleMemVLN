"""Serving-persistent GeoAddress architecture contract, separate from teacher metadata."""
import math


def parse_geoaddress(config):
    value = config['model'].get('geoaddress')
    if value is None:
        return None
    expected = dict(version='geoaddress_v1', layer=24, rank=8, heads=[0, 1], init_seed=429)
    if value != expected:
        raise ValueError('GeoAddress requires the exact v1 layer24/rank8/raw-heads0,1/seed429 contract')
    if (config['model']['output_mode'] != 'qwen_text' or config['memory']['mode'] != 'window8'
            or config['model'].get('step_lane', {}).get('enabled', False)):
        raise ValueError('GeoAddress requires native Window8 text without lanes')
    return dict(value)
