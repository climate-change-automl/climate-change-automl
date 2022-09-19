import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

def is_pareto_efficient_simple(costs):
    """
    Plot the Pareto Front in our 2d example.

    source from: https://stackoverflow.com/a/40239615
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)

            # And keep self
            is_efficient[i] = True
    return is_efficient


def plot_pareto_from_runhistory(observations):
    """
    This is only an example function for 2d plotting, when both objectives
    are to be minimized
    """

    # find the pareto front
    efficient_mask = is_pareto_efficient_simple(observations)
    front = observations[efficient_mask]
    # observations = observations[np.invert(efficient_mask)]

    obs1, obs2 = observations[:, 0], observations[:, 1]
    front = front[front[:, 0].argsort()]

    # add the bounds
    x_upper = np.max(obs1)
    y_upper = np.max(obs2)
    front = np.vstack([[front[0][0], y_upper], front, [x_upper, np.min(front[:, 1])]])

    x_front, y_front = front[:, 0], front[:, 1]

    plt.scatter(obs1, obs2)
    plt.step(x_front, y_front, where="post", linestyle=":")
    plt.title("Pareto-Front")

    plt.xlabel("Cost")
    plt.ylabel("Time")
    plt.savefig("cost_pareto.jpg")

def main():
    logger = logging.getLogger("my-logger")

    restore_state = './smac3-output_2022-09-15/run_1608637542'
    rh_path = os.path.join(restore_state, "runhistory.json")
    stats_path = os.path.join(restore_state, "stats.json")
    traj_path_aclib = os.path.join(restore_state, "traj_aclib2.json")
    traj_path_old = os.path.join(restore_state, "traj_old.csv")
    scenario_path = os.path.join(restore_state, "scenario.txt")
    if not os.path.isdir(restore_state):
        raise FileNotFoundError("Could not find folder from which to restore.")
    
    scen = Scenario(scenario=scenario_path)
    rh = RunHistory()
    rh.load_json(rh_path, scen.cs)
    logger.debug("Restored runhistory from %s", rh_path)
    stats = Stats(scen)
    stats.load(stats_path)
    logger.debug("Restored stats from %s", stats_path)
    with open(traj_path_aclib, "r") as traj_fn:
        traj_list_aclib = traj_fn.readlines()
    with open(traj_path_old, "r") as traj_fn:
        traj_list_old = traj_fn.readlines()
    cost = np.vstack([v[0] for v in rh.data.values()])
    plot_pareto_from_runhistory(cost)

if __name__ == '__main__':
    main()