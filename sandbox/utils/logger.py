from sandbox import _package_dir
import sys
import logging
import logging.config
from typing import Optional, Dict
from colorama import Fore, Back, Style

# Record the logger of all the packages for debugging and error handling
verbose = False
if verbose:
    logging.basicConfig(filename=_package_dir+"/../main.log",
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                        #datefmt='%Y/%m/%d %I:%M:%S %p'
                        )

class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


formatter = ColoredFormatter(
    '{color}{name:10}: {levelname}{reset} | {message}',
    style='{',
    # datefmt='%Y-%m-%d %H:%M:%S',
    colors={
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
)

# Get the root formatter
logger = logging.getLogger("sandbox")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(_package_dir+'.log', mode='w')
frm = logging.Formatter('%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s')
fh.setFormatter(frm)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


# create console handler
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
console.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(console)


def set_logger(name, level = logging.DEBUG):
    """
    Create a new handle in any destination
    Args:
        name: name of new handle
    Returns:
    """
    if name[:8] != "sandbox.":
        name = "sandbox."+name
    if not logging.getLogger(name).hasHandlers():
        logging.getLogger(name).addHandler(name)

    log = logging.getLogger(name)
    log.setLevel(level)
    return log

def set_level():
    pass
