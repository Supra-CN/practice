from pathlib import Path

from src import path_utils

project_root_dir = Path(__file__).parent.parent

work_space_root_dir = project_root_dir

gen_out_dir = path_utils.require_dir(Path(project_root_dir, 'gen_out'))

