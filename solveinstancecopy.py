import mrcpsp
from benders import Benders 


def solve(nominal_data_file, Gamma, time_limit,cost,e_over, max_durational_deviations=None, print_log=False):
    """
    Solve instance given in nominal_data_file with Gamma using Benders' decomposition. 
    Max durational deviations can be explicitly provided for each activity in the input, but if not provided
    they are set to floor(0.7 * nominal values). This function is used to test code for specific instances.

    :param nominal_data_file: Absolute file path to deterministic MRCPSP instance from which to create robust instance.
    :type nominal_data_file: str
    :param Gamma: Robustness parameter, i.e. max number of jobs that can achieve worst-case durations simultaneously.
        Takes integer value from 0 to n (= number of non-dummy jobs in instance).
    :type Gamma: int
    :param time_limit: Time limit (in seconds) to give to chosen solution method.
    :type time_limit: int
    :param max_durational_deviations: Optional parameter. Dictionary with max durational deviation for each mode of each
        activity.
    :type max_durational_deviations: dict
    :param print_log: Indicates whether or not to print solve log to terminal. Defaults to False.
    :type print_log: bool
    """
    # load instance
    instance = mrcpsp.load_nominal_mrcpsp(nominal_data_file)
    if max_durational_deviations:
        instance.set_dbar_explicitly(max_durational_deviations)
    else:
        instance.set_dbar_uncertainty_level(0.7)

    # solve instance using Benders' decomposition
    print("Solving {} using Benders' decomposition:".format(instance.name))
    print('"""""""""""""""""""""""""""""""""""""""""""""\n')
 

    benders_sol = Benders(instance, Gamma, time_limit, cost=cost, e_over=e_over).solve(print_log=print_log)
    print("objval:", benders_sol['objval'])
    print("runtime:", benders_sol['runtime'])
    print("n_iterations:", benders_sol['n_iterations'])
    print("modes:", benders_sol['modes'])
    print("network:", benders_sol['network'])
    print("resource flows:", benders_sol['flows'])


if __name__ == "__main__":
 

    test_instance = 'mrcpsp_toy_example.mm'
    deviations ={0:[0],1:[1,1],2:[1,3,1],3:[0,1],4:[2,1],5:[1,1],6:[1,1],7:[0]}
    cost = [[0],[0,0],[0,0,0],[0,0],[0,0],[0,0],[0,0],[0]]
    e_over = 1
    solve(test_instance, 2, 60, max_durational_deviations=deviations, cost=cost,e_over=e_over,print_log=False)
