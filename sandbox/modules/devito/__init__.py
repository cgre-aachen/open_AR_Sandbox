from sandbox import set_logger
logger = set_logger(__name__)
try:
    from .seismic_sandbox import SeismicModule
except:
    logger.warning("Devito dependencies not installed")