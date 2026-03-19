from .bce import build_loss as build_bce

def get_loss(loss_name='bce', num_classes=1):
    return build_bce()