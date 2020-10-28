from warnings import warn
try:
    from .gempy_module import GemPyModule
except:
    warn("Gempy module will not work. Gempy dependencies not found")


if __name__ == '__main__':
    pass