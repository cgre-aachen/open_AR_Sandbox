from warnings import warn
try:
    from .geoelectrics import GeoelectricsModule
except:
    warn("Geophysics module will not work. PyGimli dependencies not found")


if __name__ == '__main__':
    pass