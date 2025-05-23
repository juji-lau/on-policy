from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args, reward_type="individual"):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
        # NEW:
        reward_type      :   TODO
        # NEW END
        
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # NEW
    # Inject reward_type into scenario
    scenario.reward_type = reward_type 
    # NEW END:
    # create world
    world = scenario.make_world(args)
    
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    # NEW:
    # Store reward_type in the env object too (optional but helpful)
    env.reward_type = reward_type
    # NEW END
    return env
