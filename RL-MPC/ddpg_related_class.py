import gym
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers, activations
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import data_file


try:
    tf.enable_eager_execution()

except:
    pass

class unit_sphere_projection_layer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(unit_sphere_projection_layer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    pass
    '''self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])'''

  def call(self, inputs):
    #print("********---------", tf.math.l2_normalize(inputs).shape)
    return tf.math.l2_normalize(inputs)
    #tf.math.sqrt(tf.math.add_n(tf.math.square(inputs)))
    #tf.matmul(inputs, self.kernel)



class DDPG:

    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0, polyak_factor=0, buff_size=1000, samp_size=64):
        #num_veh = 11
        self.num_states = data_file.num_features*data_file.num_veh#data_file.max_vehi_per_lane*data_file.lane_max#env.observation_space.shape[0]
        #print("Size of State Space ->  {}".format(self.num_states))
        
                
        #self.num_actions = (data_file.num_dem_param + 2)*data_file.num_veh#data_file.max_vehi_per_lane*data_file.lane_max
        self.num_actions = (2)*data_file.num_veh#data_file.max_vehi_per_lane*data_file.lane_max

        
        #print("Size of Action Space ->  {}".format(self.num_actions))

        #upper_bound = env.action_space.high[0]
        #lower_bound = env.action_space.low[0]

        #print("Max Value of Action ->  {}".format(upper_bound))
        #print("Min Value of Action ->  {}".format(lower_bound))

        self.noise_std_dev = noise_std_dev

        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))
        #self.ou_noise_pretc  = OUActionNoise(mean=np.zeros(self.num_actions_pretc ), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions_pretc))


        self.actor_model_ = self.get_actor()
        self.critic_model_ = self.get_critic()

        '''self.actor_model_.save_weights(f"./data/snapshot_data/network_init_weights/actor_sim_{sim}")
        self.critic_model_.save_weights(f"./data/snapshot_data/network_init_weights/critic_sim_{sim}")'''

        '''self.actor_model_.load_weights(f"./data/snapshot_data/network_init_weights/actor_sim_{sim}")
        self.critic_model_.load_weights(f"./data/snapshot_data/network_init_weights/critic_sim_{sim}")'''

        '''self.actor_model_.load_weights(f"./data/trained_weights/actor_itr_{sim*1000}")
        self.critic_model_.load_weights(f"./data/trained_weights/critic_itr_{sim*1000}")'''

        self.target_actor_ = self.get_actor()
        self.target_critic_ = self.get_critic()

        # Making the weights equal initially
        self.target_actor_.set_weights(self.actor_model_.get_weights())
        self.target_critic_.set_weights(self.critic_model_.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = cri_lr
        self.actor_lr = act_lr

        self.critic_optimizer_ = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer_ = tf.keras.optimizers.Adam(self.actor_lr)

        #total_episodes = 250
        # Discount factor for future rewards
        self.gamma_ = disc_factor
        # Used to update target networks
        self.tau_ = polyak_factor

        self.buff_size = buff_size
        self.samp_size = samp_size

        self.buffer = Buffer(buffer_capacity=self.buff_size, batch_size=self.samp_size, state_size=self.num_states, action_size=self.num_actions, actor_m=self.actor_model_, critic_m=self.critic_model_, tar_act_m=self.target_actor_, tar_cri_m=self.target_critic_, gamma=self.gamma_, tau=self.tau_, cri_optimizer=self.critic_optimizer_, act_optimizer=self.actor_optimizer_)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003) # random initailisation between -/+0.003| initial weight for NN

        inputs = layers.Input(shape=(self.num_states,1)) # defines the input layer shape (self.num_states,1), # num_states of vector 

        '''layer_1 = layers.Flatten() (inputs)

        layer_1 = layers.Dense(4*self.num_actions, activation='relu') (layer_1)

        layer_2 = layers.Dense(2*self.num_actions) (layer_1)

        layer_3 = layers.Dense(self.num_actions) (layer_2)'''

        #bn_inputs = layers.BatchNormalization() (inputs)

        #inputs = layers.Reshape(target_shape=(self.num_states,1)) (inputs)
        layer_1_dem = layers.Conv1D(32, data_file.num_features, strides=data_file.num_features, activation="relu") (inputs)  #32 filters each of size num_features
        #layers.Conv1D outputs a feature map
        layer_1_dem = layers.Flatten() (layer_1_dem)
        layer_1_dem = layers.Reshape(target_shape=(layer_1_dem.shape[-1], 1)) (layer_1_dem)
        layer_2_dem = layers.Conv1D(64, kernel_size=32, strides=32) (layer_1_dem)  #kernel_size -- filter shape
        layer_2_dem = layers.Flatten() (layer_2_dem)
        layer_2_dem = layers.Reshape(target_shape=(layer_2_dem.shape[-1], 1)) (layer_2_dem)

        layer_1 = layers.Conv1D(16, 64, strides=64, activation="relu") (layer_2_dem)
        layer_1 = layers.Flatten() (layer_1)
        layer_1 = layers.Reshape(target_shape=(layer_1.shape[-1], 1)) (layer_1)
        layer_2 = layers.Conv1D(4, kernel_size=16, strides=16) (layer_1)
        layer_2 = layers.Flatten() (layer_2)
        layer_2 = layers.Reshape(target_shape=(layer_2.shape[-1], 1)) (layer_2)
        layer_3 = layers.Conv1D(1, 4, strides=4, kernel_initializer=last_init, bias_initializer=last_init) (layer_2)
        layer_3 = layers.Flatten() (layer_3)
        outputs_pi = activations.softmax(layer_3) #unit_sphere_projection_layer(int(self.num_actions)) (layer_3)


        #for demand
        #layer_3_dem = layers.Conv1D((data_file.num_dem_param), kernel_size=64, strides=64, kernel_initializer=last_init, bias_initializer=last_init) (layer_2_dem)
        #outputs_di = layers.Flatten() (layer_3_dem)



        # for TC
        layer_3_tc = layers.Conv1D(16, kernel_size=64, strides=64, kernel_initializer=last_init, bias_initializer=last_init, activation="relu") (layer_2_dem)
        layer_3_tc = layers.Flatten() (layer_3_tc)
        layer_3_tc = layers.Reshape(target_shape=(layer_3_tc.shape[-1], 1)) (layer_3_tc)
        layer_4_tc = layers.Conv1D(8, kernel_size=16, strides=16, kernel_initializer=last_init, bias_initializer=last_init, activation="relu") (layer_3_tc)
        layer_4_tc = layers.Flatten() (layer_4_tc)
        layer_4_tc = layers.Reshape(target_shape=(layer_4_tc.shape[-1], 1)) (layer_4_tc)
        layer_5_tc = layers.Conv1D(1, kernel_size=8, strides=8, kernel_initializer=last_init, bias_initializer=last_init) (layer_4_tc)
        outputs_tc = layers.Flatten() (layer_5_tc)

        # outputs = []

        # for ind_ in range(self.num_actions):
        #     out_ = outputs_[ind_]
        #     if not (ind_%(data_file.num_dem_param+1)):
        #         outputs.append(activations.softmax(\
        #         tf.convert_to_tensor([out_])))

        #     else:
        #         outputs.append(tf.convert_to_tensor([out_]))

        #outputs = layers.Concatenate() ([outputs_pi, outputs_di])
#        outputs = layers.Concatenate() ([outputs_pi, outputs_di, outputs_tc])
        outputs = layers.Concatenate() ([outputs_pi, outputs_tc])

        model = tf.keras.Model(inputs, outputs)


        plot_model(model, to_file='model.png', show_shapes=True)
        #print(model.summary())
        return model


    def get_critic(self):

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # State as input
        state_input = layers.Input(shape=(self.num_states))
        bn_state_input = layers.BatchNormalization() (state_input)


        state_out = layers.Dense(512, activation="relu")(bn_state_input)
        bn_state_input = layers.BatchNormalization() (state_out)

        #print("******* critic state layer shape:", state_out.shape)
        state_out = layers.Dense(128, activation="relu")(bn_state_input)
        bn_state_input = layers.BatchNormalization() (state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(128, activation="relu")(action_input)
        bn_action_out = layers.BatchNormalization() (action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([bn_state_input, bn_action_out])

        out = layers.Dense(256, activation="relu")(concat)
        #print("******* critic out layer shape:", out.shape)
        out = layers.Dense(256, activation="relu")(out)
        critic_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], critic_out)


        #print(model.summary())

        return model


    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model_(state))
        noise = noise_object()
        #print(len(noise))

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise #np.maximum(noise, 0)

        '''rand = np.random.uniform(0,1)

        if rand < 0.05:
            sampled_actions = sampled_actions + np.random.uniform(low=1, high=25, size=np.shape(sampled_actions))'''
        
        # We make sure action is within bounds
        legal_action = sampled_actions #np.clip(sampled_actions, lower_bound, upper_bound)

        #print("legal_action:", legal_action)

        return [np.squeeze(legal_action)]

    

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=5e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.t = 0
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        self.t += self.dt
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.random.normal(size=self.mean.shape) * (1/self.t)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, state_size=12, action_size=5, actor_m=None, critic_m=None, tar_act_m=None, tar_cri_m=None, gamma=0.99, tau=0.001, cri_optimizer=None, act_optimizer=None):
        # Number of "experiences" to store at max
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.actor_model = actor_m
        self.critic_model = critic_m
        self.target_actor = tar_act_m
        self.target_critic = tar_cri_m
        self.gamma = gamma
        self.tau = tau
        self.critic_optimizer = cri_optimizer
        self.actor_optimizer = act_optimizer

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size))
        

    # Takes (s,a,r,s') obervation tuple as input
    def remember(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    #@tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        state_action_batch = [None for _ in range(self.batch_size)] #np.zeros((self.batch_size,1))

        #alksjdf = tf.constant([1,2,3])

        #print("type dummy:", type(alksjdf))



        '''for sa_index in range(self.batch_size):

            for robo_num in range(data_file.num_veh):
                #print(state_batch[sa_index], action_batch[sa_index], "********")

                #print("shape****:", state_batch[sa_index][data_file.num_features*robo_num : data_file.num_features*(robo_num+1)], tf.reshape(action_batch[sa_index][robo_num], [1]))

                #print("shape****:", tf.shape(tf.transpose(state_batch[sa_index])), tf.shape(action_batch[sa_index]))
                #x = tf.cast(state_batch[sa_index], tf.float64)
                #tf.cast(action_batch[0][sa_index], tf.int32)
                #print("type:", type(state_batch[sa_index]), type(action_batch[0][sa_index]))
                if robo_num == 0:
                    #state_action_batch[sa_index][(data_file.num_features+1)*robo_num : (data_file.num_features+1)*(robo_num+1)] = tf.concat((state_batch[sa_index][data_file.num_features*robo_num : data_file.num_features*(robo_num+1)], tf.reshape(action_batch[sa_index][robo_num], [int(self.action_size/data_file.num_veh)])), axis=-1)
                    state_action_batch[sa_index] = tf.concat((state_batch[sa_index][data_file.num_features*robo_num : data_file.num_features*(robo_num+1)], tf.reshape(action_batch[sa_index][robo_num], [int(self.action_size/data_file.num_veh)])), axis=-1)
                
                else:
                    #state_action_batch[sa_index][(data_file.num_features+1)*robo_num : (data_file.num_features+1)*(robo_num+1)] = tf.concat((state_batch[sa_index][data_file.num_features*robo_num : data_file.num_features*(robo_num+1)], tf.reshape(action_batch[sa_index][robo_num], [int(self.action_size/data_file.num_veh)])), axis=-1)
                    state_action_batch[sa_index] = tf.concat((state_action_batch[sa_index], state_batch[sa_index][data_file.num_features*robo_num : data_file.num_features*(robo_num+1)], tf.reshape(action_batch[sa_index][robo_num], [int(self.action_size/data_file.num_veh)])), axis=-1)


                #print(state_action_batch[sa_index], "************************")
                #print(a)

            state_action_batch[sa_index] = np.asarray(np.reshape(state_action_batch[sa_index], (self.state_size + self.action_size)))

            print(sa_index, "**********", state_action_batch[sa_index])

        state_action_batch = np.asarray(state_action_batch)
        print("shape*******:", np.shape(np.asarray(state_action_batch)))'''

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            # print("**---------*********-----******", next_state_batch.shape, target_actions.shape, reward_batch.shape, "\n")
            # print(f"\n {self.target_critic([next_state_batch, target_actions])[0].shape}")
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            #critic_value = self.critic_model(state_action_batch, training=True)
            #print(critic_value)
            #print("********", critic_value)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)

            '''state_action_batch = np.zeros((self.batch_size, self.state_size + self.action_size))

            for sa_index in range(self.batch_size):
                for robo_num in range(data_file.num_veh):
                    state_action_batch[sa_index][(data_file.num_features+1)*robo_num : (data_file.num_features+1)*(robo_num+1)] = tf.concat((state_batch[sa_index][data_file.num_features*robo_num : data_file.num_features*(robo_num+1)], tf.reshape(action[sa_index][robo_num], [int(self.action_size/data_file.num_veh)])), axis=-1)
                state_action_batch[sa_index] = np.reshape(state_action_batch[sa_index], (1, self.state_size + self.action_size))'''

            critic_value = self.critic_model([state_batch, actions], training=True)
            #critic_value = self.critic_model(state_action_batch, training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)#, replace=False)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


    


'''


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

rl_ret_collection = []
comb_opt_test = []
moving_avg_ret = []
ddswa_comp_ret = []
comb_opt_ret_collection = []
rl_explore_data = []

list_of_snapshots = []
time_of_capture = []
list_arrival_rate = []

curr_state_list = []
curr_action_list = []

learning_flag = 1

algo_option = "rl_modified_ddswa"

for c in os.listdir("./captured_snaps/"):
    try:
        file = open(f"./captured_snaps/{c}",'rb')
        object_file = pickle.load(file)
        file.close()
        list_of_snapshots.append(copy.deepcopy(object_file[0]))
        time_of_capture.append(copy.deepcopy(object_file[1]))
        list_arrival_rate.append(copy.deepcopy(object_file[2]))

    except Exception as e:
          print("Exception:", e)

for mean_arr, time, snapshot in zip(list_arrival_rate, time_of_capture, list_of_snapshots):

    num_of_prov_veh = functions.get_num_of_objects(snapshot.prov_veh)
    num_of_coord_veh = functions.get_num_of_objects(snapshot.coord_veh)

    snap_rl_prov_veh = copy.deepcopy(snapshot.prov_veh)
    snap_rl_coord_veh = copy.deepcopy(snapshot.coord_veh)

    
    k = 0

    # Takes about 4 min to train
    for ep in range(total_episodes):

        

        

        #prev_state = #env.reset()
        episodic_reward = 0

        

        while True:
        #for k in range(10):

            prov_veh_in_for_loop = copy.deepcopy(snap_rl_prov_veh)
            coord_veh_in_for_loop = copy.deepcopy(snap_rl_coord_veh)

            num_of_veh = functions.get_num_of_objects(prov_veh_in_for_loop)
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            #tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            #action = policy(tf_prev_state, ou_noise)
            # Recieve state and reward from environment.
            _, _, reward, state, action, _, _ = coord_phase.coord_algo(time, prov_veh_in_for_loop, coord_veh_in_for_loop, algo_option, actor_model, learning_flag, 0, ou_noise)#env.step(action)

            reward = reward/num_of_veh

            #done = 0
            #if k == 9:
            done = 1

            #k += 1



            buffer.remember((state, action, reward, state))
            episodic_reward += reward

            for itr in range(1):
                buffer.learn()
                update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

            # End this episode when `done` is True
            if done:
                break

            #prev_state = state

        ep_reward_list.append(episodic_reward)

        if (abs((max(ep_reward_list[-10:]) - min(ep_reward_list[-10:])) / np.mean(ep_reward_list[-10:])) < 0.001) and (k <= 3):
            ou_noise.reset()
            k += 1

        # Mean of last 40 episodes
        avg_reward = episodic_reward#np.mean(ep_reward_list)
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        print(k)
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()

'''