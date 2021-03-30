from sandbox import set_logger
logger = set_logger(__name__)
try:
    from .pynoddy_module import PynoddyModule
except:
    logger.warning("Pynoddy module will not work. Dependencies not found")

if __name__ == '__main__':
    pass