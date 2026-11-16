#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRUNet — emplacement conservé pour compatibilité.

Depuis la version 0.3.0, DRUNet est fourni par eoqual-neural ; voir
``eoqual.denoising.backends.neural`` (extra ``neural``).
"""
from __future__ import annotations

from ..neural import drunet_denoise

__all__ = ["drunet_denoise"]
