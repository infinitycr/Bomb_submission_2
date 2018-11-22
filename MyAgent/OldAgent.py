from pommerman.agents import *
import tensorflow as tf
import numpy as np

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def featurize(obs):
    board = np.array(obs['board'],dtype=np.float32).reshape(-1)
    bomb_blast_strength = np.array(obs['bomb_blast_strength'],dtype=np.float32).reshape(-1)
    bomb_life = np.array(obs['bomb_life'], dtype=np.float32).reshape(-1)
    position = np.array(obs['position']).astype(np.float32)
    ammo = np.array([obs['ammo']]).astype(np.float32)
    blast_strength = np.array([obs['blast_strength']]).astype(np.float32)
    can_kick = np.array([obs['can_kick']]).astype(np.float32)

    teammate = obs['teammate']
    # if teammate is not None:
    #     teammate = teammate.value
    # else:
    #     teammate = -1
    teammate = np.array([teammate]).astype(np.float32)

    enemies = obs['enemies']
    enemies = [e for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1] * (3 - len(enemies))
    enemies = np.array(enemies).astype(np.float32)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

def available_action(state):
    # 可行动的动作为0，不可行动的动作为-9999
    position = state['position']
    board = state['board']
    x = position[0]
    y = position[1]
    if state['can_kick']:
        avail_path = [0,3,6,7,8]
    else:
        avail_path = [0,6,7, 8]
    action = [0, -9999, -9999, -9999, -9999, 0]
    if state['ammo'] == 0:
        action[-1] = -9999
    if (x - 1) >= 0:
        if board[x-1,y] in avail_path:
            #可以往上边走
            action[1] = 0
    if (x + 1) <= 10:
        if board[x+1, y] in avail_path:
            #可以往下边走
            action[2] = 0
    if (y - 1) >= 0:
        if board[x,y-1] in avail_path:
            #可以往坐边走
            action[3] = 0
    if (y + 1) <= 10:
        if board[x,y+1] in avail_path:
            #可以往右边走
            action[4] = 0
    return action

class NetAgent(BaseAgent):
    def __init__(self, sess, params):
        super(NetAgent, self).__init__()
        act_dim = 6
        P_dim = 372
        with tf.variable_scope('net'):
            activation = tf.nn.relu
            self.available_moves = tf.placeholder(tf.float32, [None, act_dim], name='availableActions')
            with tf.variable_scope('policy_net'):
                self.X_ob = tf.placeholder(tf.float32, [None, P_dim], name="input")
                with tf.variable_scope('fc1'):
                    # nin = self.X_ob.shape()[1].value
                    w1 = tf.get_variable('w1', [372, 64], initializer=ortho_init(np.sqrt(2)))
                    b1 = tf.get_variable('b1', [64], initializer=tf.constant_initializer(0))
                    z1 = activation(tf.matmul(self.X_ob, w1) + b1)
                with tf.variable_scope('fc2'):
                    # nin = z1.get_shape()[1].value
                    w2 = tf.get_variable('w2', [64, 64], initializer=ortho_init(np.sqrt(2)))
                    b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0))
                    z2 = activation(tf.matmul(z1, w2) + b2)

                with tf.variable_scope('fc3_p'):
                    # nin = z2.get_shape()[1].value
                    w3 = tf.get_variable('w3', [64, act_dim], initializer=ortho_init(np.sqrt(2)))
                    b3 = tf.get_variable('b3', [act_dim], initializer=tf.constant_initializer(0))
                    self.pi = tf.matmul(z2, w3) + b3
        # self.availPi = tf.add(self.pi, self.available_moves)
        self.dist = tf.distributions.Categorical(logits=self.pi)
        self.action = self.dist.sample()
        self.sess = sess
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
        self.loadParams(params)

    def act(self, obs, action_space):
        #featurize obs
        obs_input = featurize(obs).reshape(-1, 372)
        # availacs = np.array(available_action(obs),dtype=int).reshape(1,6)
        action = self.step_policy(obs_input)
        action = np.int(action)
        return action


    def step_policy(self, obs):
        action = self.sess.run(self.action, {self.X_ob: obs})
        return action

    def loadParams(self,paramsToLoad):
        replace_target_op = [tf.assign(t, e) for t, e in zip(self.params, paramsToLoad)]
        self.sess.run(replace_target_op)
