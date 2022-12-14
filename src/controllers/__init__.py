REGISTRY = {}

from .basic_controller import BasicMAC
from .rode_controller import RODEMAC
from .role_controller import ROLEMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['rode_mac'] = RODEMAC
REGISTRY['role_mac'] = ROLEMAC
