from sandbox import set_logger
logger = set_logger(__name__)
try:
    from .geoelectrics import GeoelectricsModule
except:
    logger.warning("Geophysics module will not work. PyGimli dependencies not found")


if __name__ == '__main__':
    pass