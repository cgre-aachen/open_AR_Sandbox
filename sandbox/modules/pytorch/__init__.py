from sandbox import set_logger
logger = set_logger(__name__)
try:
    from .landscape_generation import LandscapeGeneration
except:
    logger.warning("LandscapeGeneration module will not work. Dependencies not installed")