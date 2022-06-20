# PPO learn (tf2 subclassing API version)
# coded by St.Watermelon

# 필요한 패키지 임포트
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

import metaworld
import random
import cv2

# unset LD_PRELOAD

# Conv2D -> BatchNormalization -> Activation -> Maxpooling (-> Dropout)
class ConvBatchNormMaxpool(tf.keras.layers.Layer):
    def __init__(
        self, 
        conv2d_filters, 
        conv2d_kernel_size, 
        conv2d_strides, 
        conv2d_padding,
        maxpool2d_pool_size,
        maxpool2d_strides,
        maxpool2d_padding
    ):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(
            filters = conv2d_filters,
            kernel_size = conv2d_kernel_size,
            strides = conv2d_strides,
            padding = conv2d_padding
        )
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.maxpool2d = tf.keras.layers.MaxPool2D(
            pool_size = maxpool2d_pool_size,
            strides = maxpool2d_strides,
            padding = maxpool2d_padding
        )

    def call(self, input):
        """Conv2D -> BatchNormalization -> Activation -> Maxpooling (-> Dropout)"""
        x = self.conv2d(input)
        x = self.batchnorm(x)
        x = tf.keras.activations.relu(x)
        out = self.maxpool2d(x)
        return out

## PPO 액터 신경망
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        self.CNN1 = ConvBatchNormMaxpool(
            32, 
            [3, 3], 
            [1, 1], 
            "same",
            [2, 2],
            [2, 2],
            "valid"
        )
        
        self.CNN2 = ConvBatchNormMaxpool(
            64, 
            [3, 3], 
            [1, 1], 
            "valid",
            [2, 2],
            [1, 1],
            "valid"
        )
        
        self.CNN3 = ConvBatchNormMaxpool(
            32, 
            [5, 5], 
            [2, 2], 
            "valid",
            [3, 3],
            [2, 2],
            "valid"
        )
        
        self.flatten = tf.keras.layers.Flatten()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')

    def call(self, input_img):
        # x = tf.expand_dims(input_img, -1)
        x = self.CNN1(input_img)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.flatten(x)
        
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        # 평균값을 [-action_bound, action_bound] 범위로 조정
        mu = Lambda(lambda x: x*self.action_bound)(mu)

        return [mu, std]


## PPO 크리틱 신경망
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()
        
        self.CNN1 = ConvBatchNormMaxpool(
            32, 
            [3, 3], 
            [1, 1], 
            "same",
            [2, 2],
            [2, 2],
            "valid"
        )
        
        self.CNN2 = ConvBatchNormMaxpool(
            64, 
            [3, 3], 
            [1, 1], 
            "same",
            [2, 2],
            [1, 1],
            "valid"
        )
        
        self.CNN3 = ConvBatchNormMaxpool(
            32, 
            [5, 5], 
            [2, 2], 
            "valid",
            [3, 3],
            [2, 2],
            "valid"
        )
        
        self.flatten = tf.keras.layers.Flatten()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v = Dense(1, activation='linear')

    def call(self, input_img):
        # x = tf.expand_dims(input_img, -1)
        x = self.CNN1(input_img)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.flatten(x)
        
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)
        return v


## PPO 에이전트 클래스
class PPOagent(object):

    def __init__(self, env, batch_size, actor_lr, critic_lr, ratio_clipping, max_path_len, epoch):

        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.GAE_LAMBDA = 0.9
        
        self.BATCH_SIZE = batch_size
        self.ACTOR_LEARNING_RATE = actor_lr
        self.CRITIC_LEARNING_RATE = critic_lr
        self.RATIO_CLIPPING = ratio_clipping
        self.EPOCHS = epoch
        
        self.max_path_len = max_path_len

        # 환경
        self.env = env
        # 상태변수 차원
        # self.state_dim = env.observation_space.shape[0]
        # self.state_dim = 300 * 400 * 4
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기
        self.action_bound = env.action_space.high[0]
        # 표준편차의 최솟값과 최댓값 설정
        self.std_bound = [1e-3, 1]

        # 액터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        self.actor.build(input_shape=(None, 300, 400, 16))
        self.critic.build(input_shape=(None, 300, 400, 16))

        self.actor.summary()
        self.critic.summary()

        # 옵티마이저
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []
        
        self.max_epi = 1356


    ## 로그-정책 확률밀도함수 계산
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)


    ## 액터 신경망으로 정책의 평균, 표준편차를 계산하고 행동 샘플링
    def get_policy_action(self, state):
        mu_a, std_a = self.actor(state)
        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return mu_a, std_a, action


    ## GAE와 시간차 타깃 계산
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets


    ## 배치에 저장된 데이터 추출
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack


    ## 액터 신경망 학습
    def actor_learn(self, log_old_policy_pdf, states, actions, gaes):

        with tf.GradientTape() as tape:
            # 현재 정책 확률밀도함수
            mu_a, std_a = self.actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

            # 현재와 이전 정책 비율
            ratio = tf.exp(log_policy_pdf - log_old_policy_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            loss = tf.reduce_mean(surrogate)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## 크리틱 신경망 학습
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_hat-td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_weights(path + 'actor.h5')
        self.critic.load_weights(path + 'critic.h5')


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 배치 초기화
        batch_state, batch_action, batch_reward = [], [], []
        batch_log_old_policy_pdf = []

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            _ = self.env.reset()
            state = np.zeros(shape=(300, 400, 16), dtype=np.uint8)

            # while not done:
            for _ in range(self.max_path_len):
                # 이전 정책의 평균, 표준편차를 계산하고 행동 샘플링
                mu_old, std_old, action = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 이전 정책의 로그 확률밀도함수 계산
                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)
                # We don't use original state
                _, reward_scalar, done, _ = self.env.step(action)
                reward_scalar *= 10
                img_list = []
                for cam in camera:
                    if cam in ['corner', 'corner2', 'corner3']:
                        flip = False
                    else:
                        flip = True
                        
                    img_color, depth = env.sim.render(*resolution, mode='offscreen', camera_name=cam, depth = True)
                    if flip: img_color = cv2.rotate(img_color, cv2.ROTATE_180); depth = cv2.rotate(depth, cv2.ROTATE_180)
                    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                    imgHsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
                        
                    imgMask1 = cv2.inRange(imgHsv, lb1, ub1)
                    imgMask2 = cv2.inRange(imgHsv, lb2, ub2)
                    imgMask3 = cv2.inRange(imgHsv, lb3, ub3)
                    imgMask = imgMask1 | imgMask2 | imgMask3
                    img_result = cv2.bitwise_and(img_color, img_color, mask=imgMask)
                    img_result = cv2.GaussianBlur(img_result, (0, 0), 3)
                    
                    depth = (np.max(depth)-depth) / (np.max(depth) - np.min(depth))
                    depth = np.asarray(depth * 255, dtype=np.uint8)
                    
                    depth = depth[..., np.newaxis]
                    
                    total_img = np.concatenate([depth, img_result], axis=-1) # (300, 400, 4)
                    img_list.append(total_img)
                next_state = np.concatenate(img_list, axis=-1) # (300, 400, 16)
                # shape 변환
                state = np.reshape(state, [1, 300, 400, 16])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward_scalar, [1, 1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])
                # 학습용 보상 설정
                train_reward = reward
                # 배치에 저장
                if reward_scalar != 0:
                    batch_state.append(state)
                    batch_action.append(action)
                    batch_reward.append(train_reward)
                    batch_log_old_policy_pdf.append(log_old_policy_pdf)

                # 배치가 채워질 때까지 학습하지 않고 저장만 계속
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state
                    episode_reward += reward[0]
                    print("reward: ", episode_reward)
                    time += 1
                    continue

                # 배치가 채워지면, 학습 진행
                print("***training start***")
                # 배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)
                # 배치 비움
                batch_state, batch_action, batch_reward, = [], [], []
                batch_log_old_policy_pdf = []
                # GAE와 시간차 타깃 계산
                next_v_value = self.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                gaes, y_i = self.gae_target(rewards, v_values.numpy(), next_v_value.numpy(), done)

                # 에포크만큼 반복
                for _ in range(self.EPOCHS):
                    # 액터 신경망 업데이트
                    self.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                                     tf.convert_to_tensor(states, dtype=tf.float32),
                                     tf.convert_to_tensor(actions, dtype=tf.float32),
                                     tf.convert_to_tensor(gaes, dtype=tf.float32))
                    # 크리틱 신경망 업데이트
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                # 다음 에피소드를 위한 준비
                state = next_state
                episode_reward += reward[0]
                time += 1
                
                if done:
                    print("SUCCESS")
                    self.actor.save_weights(f'save_weights/actor_{ep}_episodes_success_{_}_steps.h5')
                    self.critic.save_weights(f'save_weights/critic_{ep}_episodes_success_{_}_steps.h5')
                    np.savetxt(f'save_weights/epi_reward_{ep}_episodes_success_{_}_steps.txt', self.save_epi_reward)
                    break

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)

            if self.max_epi < episode_reward:
                self.actor.save_weights(f'save_weights/actor_{ep}_episodes.h5')
                self.critic.save_weights(f'save_weights/critic_{ep}_episodes.h5')
                np.savetxt(f'save_weights/epi_reward_{ep}_episodes.txt', self.save_epi_reward)
                print(f'save: {ep} episode / {episode_reward} reward')
                self.max_epi = episode_reward



    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.savefig('result.png')
        
        
        
        
        
# define gpus strategy
#mirrored_strategy = tf.distribute.MirroredStrategy()
        
with tf.device('/GPU:0'):
    hsv = [0, 204, 100]
    lb1 = np.array([hsv[0]-10+180, 30, 30])
    ub1 = np.array([180, 255, 255])
    lb2 = np.array([0, 30, 30])
    ub2 = np.array([hsv[0], 255, 255])
    lb3 = np.array([hsv[0], 30, 30])
    ub3 = np.array([hsv[0]+10, 255, 255])  
    
    resolution = (400, 300)
    camera = ['topview', 'corner', 'corner2', 'corner3']
    flip = True # if True, flips output image 180 degrees

    config = [
        # env, action noise pct
        ('button-press-topdown-v2', np.zeros(4)),
    ]

    max_episode_num = 10000
        
    ml1 = metaworld.ML1('button-press-topdown-v2')
    env = ml1.train_classes['button-press-topdown-v2']()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    
    
    agent = PPOagent(env, 
                     batch_size=64, 
                     actor_lr=0.002,
                     critic_lr=0.002,
                     ratio_clipping=0.1,
                     max_path_len=128,
                     epoch=10)
    
    agent.load_weights('save_weights/')
    
        
    agent.train(max_episode_num)
    
    
