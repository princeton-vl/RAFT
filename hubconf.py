dependencies = ['torch']

try:
    from hubconf_models import RAFT
except ModuleNotFoundError:
    from .hubconf_models import RAFT
