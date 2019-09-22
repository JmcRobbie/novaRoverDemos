from pkgutil import extend_path
from package import __path__
__path__ = __import__('pkgutil').extend_path(__path__, __name__)