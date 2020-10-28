from warnings import warn
try:
    from .landscape_generation import LandscapeGeneration
except:
    warn("LandscapeGeneration module will not work. Dependencies not installed")