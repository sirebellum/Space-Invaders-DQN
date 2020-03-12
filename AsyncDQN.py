#!/usr/bin/env python
import os

from processedEnv import processingWrapper
import threading
import tensorflow as tf
import random
import numpy as np
import time
import gym
import keras
from model import build_model
import io
import base64
import glob

import wandb
# initialize a new wandb run
wandb.init(project="qualcomm")
# define hyperparameters
wandb.config.episodes = 100
wandb.config.batch_size = 32
wandb.config.learning_rate = 1e-5
cumulative_reward = 0
episode = 0

#environment parameters -- Space Invaders for the Atari
experiment = "async-dqn-spaceInvaders"
game = "SpaceInvaders-v0"

#number of agents
n_agents = 8

#what to resize frames to
r_width = 84
r_height = 84

#Get size/length of the set of Actions, A.
quick = gym.make(game)
len_A = quick.action_space.n


#memory length ( action repeat of 4 as laid out in mnih et al. 5.1. "Experimental Setup" )
memory_length = 4

#as specified, n is set to 5 (both for target and local Q)
n = 3
target_n = 3

#learning rate, discount rate, annealling parameter for epsilon tuning 
alpha = wandb.config.learning_rate
gamma = 0.95

#How many time steps it will take for epsilon to decay from its ceiling to its floor.
#Controls how random the model will act, and for how long
epsilon_decay_rate = 200000


#directories & paths for storing / saving / visualization
store_dir = 'SpaceInvaders'
summary_dir = store_dir+"/summary/"
checkpoint_dir = store_dir+"/ckpts/"


#Controls env.render()
show_training = True

#parameters for saving. After this many episodes, create ckpt, or summary
e_summary = 5
e_ckpt = 500


#set this to the checkpoint to load
ckpt_to_load ="path/to/recent.ckpt"

#Number of training timesteps
T = 0


TMAX = 50000000

#values of epsilon, and the idea of sampling taken from section 5.1 
#Asynchronous Methods for Deep Reinforcement Learning, Mnih et al
def get_epsilon_floor():
    #epsilon_floor = np.array([.1,.01,.5])
    #epsilons = np.array([0.4,0.3,0.3])
    #return np.random.choice(epsilon_floor, 1, p=list(epsilons))[0]
    return 0.1

def build_graph(len_A):
    # Create shared deep q network
    s, Q = build_model(len_A, memory_length,"target-network",r_height, r_width)
    network_params = Q.trainable_weights
    q_values = Q(s)

    # Shared Q model, referred to as the "Target"

    st, target_Q = build_model(len_A, memory_length,"target-network",r_height, r_width)
    target_Q_theta = target_Q.trainable_weights
    target_q_values = target_Q(st)

    # Operation for updating target network with weights from agents local/online models
    reset_target_Q_theta = [target_Q_theta[i].assign(network_params[i]) for i in range(len(target_Q_theta))]
    
    # Cost and gradient update operation
    a = tf.placeholder("float", [None, len_A])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)

    #cost is defined by the difference between actual reward and predicted reward
    cost = tf.reduce_mean(tf.square(y - action_q_values))

    # Schedule learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(
        alpha,
        global_step,
        10,
        0.96,
        staircase=True)

    #Use an AdamOptmizer with learning rate = alpha
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params, global_step=global_step)

    return s,q_values,st,target_q_values,reset_target_Q_theta,a,y,grad_update


# Visualization setup (summaries)

def setup_summaries():

    r_e = tf.Variable(0.)
    tf.summary.scalar("Episode Reward", r_e)


    e_avg_max_Q = tf.Variable(0.)
    tf.summary.scalar("Max Q Value", e_avg_max_Q)

    epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", epsilon)

    T = tf.Variable(0.)
    var = [r_e, e_avg_max_Q, epsilon]

    temp = [tf.placeholder("float") for i in range(len(var))]


    update = [var[i].assign(temp[i]) for i in range(len(var))]
    summ = tf.summary.merge_all()

    return temp, update, summ


#implementation of a async Deep-Q learning agent with multiple threads
def async_Q_learning(thread_num, env, session,s,q_values,st,target_q_values,reset_target_Q_theta,a,y,grad_update
, len_A, summary_ops, saver):

    # Create environment using helper class processingWrapper to abstract image pre-processing for the model
    env = processingWrapper(env, memory_length,r_height,r_width,thread_num)

    #Global time
    global TMAX, T

    summary_placeholders, update_ops, summary_op = summary_ops


    # Setup network gradients

    s_batch = []
    a_batch = []
    y_batch = []


    # Define epsilon parameters
    epsilon_floor = get_epsilon_floor()

    #how randoom the model will be at beginning
    epsilon_ceiling = 1.0
    epsilon = 1.0
    epsilon_decay = (epsilon_ceiling - epsilon_floor)/epsilon_decay_rate

    print( "Starting thread ", thread_num, "with epsilon floor ", epsilon_floor)

    # spin up threads sequentially
    time.sleep(3*thread_num)

    t = 0
    while episode < wandb.config.episodes:
        # Get initial state
        # s_t -> state at t
        s_t = env.initState()

        done = False

        # Set up counters local to an instance
        # Episode reward, average max Q for the episode, episode t

        r_e = 0
        e_avg_max_Q = 0
        e_t = 0

        while True:
            # Forward pass on model, get Q(s,a) values
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            
            # Use e-greedy policy
            a_t = np.zeros([len_A])
            action_index = random.randrange(len_A) if random.random() <= epsilon else np.argmax(readout_t)


            # EPSILON DECAY FACTOR --- as the model better approximates Q*, rely on Q* more than random actions!
            # But have a floor for random actions to remain general / adaptive! 
            epsilon -= epsilon_decay if epsilon > epsilon_floor else 0
    
            # perform action
            s_t1, r_t, done = env.step(action_index)

            # Acquire+Store Gradients
            target_readout = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            if r_t == 0: clipped_r_t = 0
            elif r_t > 0: clipped_r_t = 1
            else: r_t = -1

            y_batch.append(clipped_r_t if done else clipped_r_t + gamma * np.max(target_readout) )
            a_batch.append(a_t)
            s_batch.append(s_t)
    
            # Update State and time
            s_t = s_t1
            T += 1
            t += 1

            #update episode time, and episode reward, and average episode q max
            e_t += 1
            r_e += r_t
            e_avg_max_Q += np.max(readout_t)

            # Update target network if needed
            if T % target_n == 0:
                session.run(reset_target_Q_theta)
    
            # Update online network if needed
            if t % n == 0 or done:
                if s_batch:
                    session.run(grad_update, feed_dict = {y : y_batch,
                                                          a : a_batch,
                                                          s : s_batch})
                # Reset gradients
                s_batch = []
                a_batch = []
                y_batch = []
    
            # Create ckpt if needed
            if t % e_ckpt == 0:
                saver.save(session, checkpoint_dir+"/"+experiment+".ckpt", global_step = t)
    
            # print( end of episode stats
            if done:
                stats = [r_e, e_avg_max_Q/float(e_t), epsilon]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print( " \n Thread:", thread_num, "\n T:", T, "\n t:", t, "\n Epsilon:", epsilon, "\n Reward:", r_e, "\n Q_Max: %.4f" % (e_avg_max_Q/float(e_t)), "\n % finished epsilon decay :", t/float(epsilon_decay_rate), "\n--------------------------------\n")

                if thread_num == 1:
                    # render gameplay video
                    evaluate(r_e)
                    if (episode %10 == 0):
                      mp4list = glob.glob('video/*.mp4')
                      if len(mp4list) > 0:
                        print(len(mp4list))
                        mp4 = mp4list[-1]
                        video = io.open(mp4, 'r+b').read()
                        encoded = base64.b64encode(video)
                        # log gameplay video in wandb
                        wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})

                break


def train(session, s,q_values,st,target_q_values,reset_target_Q_theta,a,y,grad_update, len_A, saver):

    # Set up game environments (one per thread)

    envs = [gym.make(game) for i in range(n_agents)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables

    session.run(tf.global_variables_initializer())

    # Initialize target network weights

    session.run(reset_target_Q_theta)

    summary_save_path = summary_dir + "/" + experiment

    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Start n_agent training threads
    async_Q_learnings = [threading.Thread(target=async_Q_learning, args=(thread_num, envs[thread_num], session, s,q_values,st,target_q_values,reset_target_Q_theta,a,y,grad_update, 
                                        len_A, summary_ops, saver)) for thread_num in range(n_agents)]
    for t in async_Q_learnings:
        t.start()

    
    # Show the agents training and write summary statistics
    last_summary_time = 0
    

    #does the rendering
    while True:
        if show_training:
            for env in envs:
                env.render()
        now = time.time()

        #determines when to write a summary file
        if now - last_summary_time > e_summary:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now

    #join threads
    for t in async_Q_learnings:
        t.join()

def evaluate(episodic_reward):
  '''
  Takes in the reward for an episode, calculates the cumulative_avg_reward
    and logs it in wandb. If episode > 100, stops logging scores to wandb.
    Called after playing each episode. See example below.

  Arguments:
    episodic_reward - reward received after playing current episode
  '''
  global episode
  global cumulative_reward
  episode += 1
  print("Episode: %d"%(episode))

  # your models will be evaluated on 100-episode average reward
  # therefore, we stop logging after 100 episodes
  if (episode > 100):
    print("Scores from episodes > 100 won't be logged in wandb.")
    return

  # log total reward received in this episode to wandb
  wandb.log({'episodic_reward': episodic_reward})

  # add reward from this episode to cumulative_reward
  cumulative_reward += episodic_reward

  # calculate the cumulative_avg_reward
  # this is the metric your models will be evaluated on
  cumulative_avg_reward = cumulative_reward/episode

  # log cumulative_avg_reward over all episodes played so far
  wandb.log({'cumulative_avg_reward': cumulative_avg_reward})

def main(_):
  g = tf.Graph()
  session = tf.Session(graph=g)
  with g.as_default(), session.as_default():
    keras.backend.set_session(session)
    s,q_values,st,target_q_values,reset_target_Q_theta,a,y,grad_update = build_graph(len_A)
    saver = tf.train.Saver()
    train(session, s,q_values,st,target_q_values,reset_target_Q_theta,a,y,grad_update, len_A, saver)

if __name__ == "__main__":
  tf.app.run()
