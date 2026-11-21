# Only the task used for real-robot deployment is registered in this release.
from .overhand import Overhand

isaacgym_task_map = {
    "Overhand": Overhand,
}
