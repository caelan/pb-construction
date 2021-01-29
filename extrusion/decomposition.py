from collections import defaultdict

from pddlstream.language.constants import get_prefix, get_function


def partition_plan(plan):
    plan_from_robot = defaultdict(list)
    for action in plan:
        plan_from_robot[action.args[0]].append(action)
    return plan_from_robot


def compute_total_orders(plan):
    # TODO: partially order trajectories
    partial_orders = set()
    for name, actions in partition_plan(plan).items():
        last_element = None
        for i, action in enumerate(actions):
            if action.name == 'print':
                r, n1, e, n2, q1, q2, t = action.args
                if last_element is not None:
                    # TODO: need level orders to synchronize between robots
                    # TODO: useful for collision checking
                    partial_orders.add((e, last_element))
                last_element = e
            else:
                raise NotImplementedError(action.name)
    return partial_orders


def extract_static_facts(plan, certificate, initial_confs):
    # TODO: use certificate instead
    # TODO: only keep objects used on the plan
    #static_facts = []
    static_facts = [f for f in certificate.all_facts if get_prefix(get_function(f))
                    in ['distance', 'trajtrajcollision']]
    for name, actions in partition_plan(plan).items():
        last_element = None
        last_conf = initial_confs[name]
        for i, action in enumerate(actions):
            if action.name == 'print':
                r, n1, e, n2, q1, q2, t = action.args
                static_facts.extend([
                    ('PrintAction',) + action.args,
                    ('Assigned', r, e),
                    ('Conf', r, q1),
                    ('Conf', r, q2),
                    ('Traj', r, t),
                    ('CTraj', r, t),
                    # (Start ?r ?n1 ?e ?q1) (End ?r ?e ?n2 ?q2)
                    ('Transition', r, q2, last_conf),
                ])
                if last_element is not None:
                    static_facts.append(('Order', e, last_element))
                last_element = e
                last_conf = q1
                # TODO: save collision information
            else:
                raise NotImplementedError(action.name)
        static_facts.extend([
            ('Transition', name, initial_confs[name], last_conf),
        ])
    return static_facts