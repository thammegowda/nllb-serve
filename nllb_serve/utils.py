import resource
from typing import Tuple
import sys

def max_RSS(who=resource.RUSAGE_SELF) -> Tuple[int, str]:
    """Gets memory usage of current process, maximum so far.
    Maximum so far, since the system call API doesnt provide "current"
    :returns (int, str)
       int is a value from getrusage().ru_maxrss
       str is human friendly value (best attempt to add right units)
    """
    mem = resource.getrusage(who).ru_maxrss
    h_mem = mem
    if 'darwin' in sys.platform:  # "man getrusage 2" says we get bytes
        h_mem /= 10 ** 3  # bytes to kilo
    unit = 'KB'
    if h_mem >= 10 ** 3:
        h_mem /= 10 ** 3  # kilo to mega
        unit = 'MB'
    return mem, f'{int(h_mem)}{unit}'