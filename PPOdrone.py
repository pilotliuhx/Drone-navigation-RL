import sys
import torch
import rospy
from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
from simenv import sim_env
from geometry_msgs.msg import Vector3

goal = Vector3()
goal.x = 3
goal.y = 0
goal.z = 1
max_step_per_eps = 400
state_dim = 4
action_dim = 2
max_action = 1

def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': 
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)
    model.learn(total_timesteps=200000000)

def test(env, actor_model):
    print(f"Testing {actor_model}", flush=True)
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)
    obs_dim = state_dim
    act_dim = action_dim
    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))
    eval_policy(policy=policy, env=env, render=True)

def main(args):
    hyperparameters = {
        'timesteps_per_batch' : 2048,
        'max_timesteps_per_episode' : max_step_per_eps,
        'gamma' : 0.99,
        'n_updates_per_iteration' : 10,
        'lr' : 3e-4,
        'clip' : 0.2,
        'obs_dim' : state_dim,
        'act_dim' : action_dim,
        'max_action' : max_action
    }
    rospy.init_node('simenv_node', anonymous=True)
    env = sim_env(goal, max_step_per_eps)
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)