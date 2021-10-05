from . import agents, environments, policies, value_functions, utils
import datetime
import os

def dict_to_dirname(x):
    return "-".join([ "{}_{}".format(k,v.__name__ if type(v) is type else v) for k,v in x.items() ])

def class_and_args_to_dirname(class_, args_):
    return "{}-{}".format(class_.__name__, dict_to_dirname(args_))

def date_dirname():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def create_agent_and_env(
        agent_class=agents.base.RandomAgent,
        agent_args={},
        env_name="GridCliff-v0",
        env_wrappers=[]
    ):
    import gym

    if "save_dir" not in agent_args.keys():
        agent_args["save_dir"] = ""
    else:
        agent_args["save_dir"]+= "-"

    e = gym.make(env_name)
    for wrapper, args in env_wrappers:
        e = wrapper(e, **args)

    agent_args["save_dir"]+= "{}-{}".format(
        date_dirname(),
        env_name + "".join(["-{}".format(w.class_name()) for w,_ in env_wrappers])
    )
    print(e)
    return agent_class(e, **agent_args)

def create_agent_from_env(env, agent_class, value_class, policy_class,
                               agent_args={}, value_args={}, policy_args={}
                          ):
    save_suffix = date_dirname() + "-" + str(env).replace("<",".").replace(">","")
    if "save_dir" not in agent_args.keys():
        agent_args["save_dir"] = "results"+os.path.sep
    else:
        agent_args["save_dir"]+= "-"
    agent_args["save_dir"]+= save_suffix

    if "infos" not in agent_args.keys():
        agent_infos = {}
    else:
        agent_infos = agent_args["infos"]
    agent_infos.update({
        'agent_class': agent_class.__name__,
        'value_class': value_class.__name__,
        'policy_class': policy_class.__name__,
        'agent_args': agent_args.copy(),
        'value_args': value_args.copy(),
        'policy_args': policy_args.copy(),
    })
    agent_args["infos"] = agent_infos

    value_args.update({
        'env': env
    })
    value_function = value_class(**value_args)

    policy_args.update({
        'value_function': value_function
    })
    policy = policy_class(**policy_args)

    agent_args.update({
        'env': env,
        'value_function': value_function,
        'policy': policy
    })

    agent = agent_class(**agent_args)
    return agent
