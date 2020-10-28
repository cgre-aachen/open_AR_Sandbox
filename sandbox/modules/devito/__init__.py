from warnings import warn
try:
    from .seismic_sandbox import SeismicModule
except:
    warn("Devito dependencies not installed")