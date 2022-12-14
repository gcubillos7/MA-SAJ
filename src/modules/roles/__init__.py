REGISTRY = {}

from .dot_role import DotRole
from .q_role import QRole
from .masaj_role import MASAJRole, MASAJRoleDiscrete
REGISTRY['dot'] = DotRole
REGISTRY['q'] = QRole
REGISTRY['msj'] = MASAJRole
REGISTRY['msj_discrete'] = MASAJRoleDiscrete
