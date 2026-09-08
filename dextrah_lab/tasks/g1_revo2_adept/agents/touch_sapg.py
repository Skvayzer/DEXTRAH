"""Reuse the proven SAPG architecture while labeling tactile runs correctly."""
from pathlib import Path
import yaml


def config():
    cfg = yaml.safe_load((Path(__file__).parent/'g1_revo2_bps128_sapg.yaml').read_text())
    cfg['params']['config']['name'] = 'g1_sapg_bps128_touch'
    cfg['wandb_group'] = 'g1-tactile-ablation'
    cfg['wandb_name'] = 'g1-sapg-bps128-touch-'
    cfg['wandb_tags'] = ['sapg','g1','revo2','bps128','tactile-option','no-fabrics','no-pca']
    return cfg
