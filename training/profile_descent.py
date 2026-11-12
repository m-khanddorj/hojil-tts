"""A common local descent direction and exact bounded-profile step conversion."""
import torch


def common_direction(voice,anchor):
    if (voice.ndim!=1 or voice.shape!=anchor.shape or not torch.isfinite(voice).all()
            or not torch.isfinite(anchor).all() or voice.norm()==0):
        raise ValueError('Invalid profile gradients')
    v,a=voice.double(),anchor.double();vn,an=v.norm(),a.norm()
    direction=-v/vn
    cosine=None
    if an>0:
        cosine=float(v.dot(a)/(vn*an));direction=direction-a/an
    if direction.norm()<1e-8:raise ValueError('Opposed objectives have no common descent')
    direction=direction/direction.norm()
    if direction.dot(v)>=0 or (an>0 and direction.dot(a)>=0):raise ValueError('No strict common descent')
    return direction.to(voice.dtype),dict(gradient_cosine=cosine,voice_norm=float(vn),anchor_norm=float(an),
        directional_voice_derivative=float(direction.dot(v)),directional_anchor_derivative=float(direction.dot(a)))


def raw_for_displacement(direction,fraction,radius_fraction):
    if (direction.ndim!=1 or not torch.isfinite(direction).all() or direction.norm()==0
            or not 0<fraction<radius_fraction<1):raise ValueError('Invalid profile displacement')
    magnitude=fraction/(radius_fraction**2-fraction**2)**.5
    return direction/direction.norm()*magnitude
