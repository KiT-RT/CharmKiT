# import umbridge
import numpy as np

from src.config_utils import read_username_from_config
from src.models.lattice import get_qois_col_names, model

from src.simulation_utils import execute_slurm_scripts, wait_for_slurm_jobs
from src.general_utils import (
    create_lattice_samples_from_param_range,
    load_lattice_samples_from_npz,
    delete_slurm_scripts,
)

from src.general_utils import parse_args


def main():
    args = parse_args()
    print(f"HPC mode = { args.use_slurm}")
    print(f"Load from npz = {args.load_from_npz}")
    print(f"HPC with singularity = { args.use_singularity}")

    hpc_operation = args.use_slurm  # Flag when using HPC cluster
    load_from_npz = args.load_from_npz
    singularity_hpc = args.use_singularity

    # print(f"Use rectangular_mesh = {args.rectangular_mesh}")
    rectangular_mesh = False

    # --- Define parameter ranges ---

    #  characteristic length of the cells:  #grid cells = O(1/cell_size^2)
    parameter_range_grid_cell_size = [0.015]

    # quadrature order (must be an even number):  #velocity grid cells = O(order^2)
    parameter_range_quad_order = [6]

    # Prescribed range for LATTICE_DSGN_ABSORPTION_BLUE
    parameter_range_abs_blue = [5, 10, 50, 100]  # default: 10
    # Prescribed range for LATTICE_DSGN_SCATTER_WHITE
    parameter_range_scatter_white = [0.1, 0.5, 1, 5, 10]  # default: 1

    if load_from_npz:  # TODO
        raise NotImplementedError
        design_params, design_param_names = load_lattice_samples_from_npz(
            "sampling/pilot-study-samples-hohlraum-05-29-24.npz"
        )
        exit("TODO")
    else:
        design_params, design_param_names = create_lattice_samples_from_param_range(
            parameter_range_grid_cell_size,
            parameter_range_quad_order,
            parameter_range_abs_blue,
            parameter_range_scatter_white,
        )

    if hpc_operation:
        print("==== Execute HPC version ====")
        directory = "./benchmarks/lattice/slurm_scripts/"

        delete_slurm_scripts(directory)  # delete existing slurm files for hohlraum
        call_models(
            design_params,
            hpc_operation_count=1,
            singularity_hpc=singularity_hpc,
            rectangular_mesh=rectangular_mesh,
        )

        user = read_username_from_config("./slurm_config.txt")
        if user:
            print("Executing slurm scripts with user " + user)
            execute_slurm_scripts(directory, user)
            wait_for_slurm_jobs(user=user, sleep_interval=10)
        else:
            print("Username could not be read from config file.")

        qois = call_models(design_params, hpc_operation_count=2)
    else:
        qois = call_models(
            design_params,
            hpc_operation_count=0,
            rectangular_mesh=rectangular_mesh,
            singularity_hpc=singularity_hpc,
        )

    print("design parameter matrix")
    print(design_param_names)
    print(design_params)
    print("quantities of interest:")
    print(get_qois_col_names())
    print(qois)
    np.savez(
        "benchmarks/lattice/sn_study_lattice.npz",
        qois=qois,
        design_params=design_params,
        qoi_column_names=get_qois_col_names(),
        design_param_column_names=design_param_names,
    )

    print("======== Finished ===========")
    return 0


def call_models(
    design_params, hpc_operation_count, singularity_hpc=True, rectangular_mesh=False
):
    qois = []
    for column in design_params:
        input = column.tolist()
        print(input)
        input.append(hpc_operation_count)
        input.append(singularity_hpc)
        input.append(rectangular_mesh)

        res = model([input])
        qois.append(res[0])

    return np.array(qois)


if __name__ == "__main__":
    main()
