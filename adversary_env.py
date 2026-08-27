import numpy as np
import time
from stable_baselines3 import DQN
from stable_baselines3 import SAC, DDPG,  PPO
from model import XGBoostIDSWrapper
from train_classifer import train_model
from stable_baselines3.common.env_util import make_vec_env
from sklearn.model_selection import train_test_split
import gym
import numpy as np
import joblib
from gym import spaces
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.callbacks import BaseCallback
import os

class ContrastiveBank:
    def __init__(self, dim, ema=0.95):
        self.dim = dim
        self.ema = ema
        self.protos = {}   # agent_id -> prototype vector
        self.counts = {}   # agent_id -> count
        self.episode_embs = {} # For plotting the agent's policies

    @staticmethod
    def _norm(v, eps=1e-8):
        n = np.linalg.norm(v) + eps
        return v / n

    def update(self, agent_id, emb):
        emb = self._norm(emb)
        if agent_id not in self.protos:
            self.protos[agent_id] = emb.copy()
            self.counts[agent_id] = 1
        else:
            self.protos[agent_id] = self.ema * self.protos[agent_id] + (1 - self.ema) * emb
            self.protos[agent_id] = self._norm(self.protos[agent_id])
            self.counts[agent_id] += 1

    def contrastive_reward(self, agent_id, emb):
        # If no other agents yet, no contrastive signal
        if len(self.protos) <= 1:
            return 0.0

        emb = self._norm(emb)
        others = [pid for pid in self.protos.keys() if pid != agent_id]
        if not others:
            return 0.0

        # similarity to others (we want this LOW)
        neg_sims = [float(np.dot(emb, self.protos[pid])) for pid in others]
        neg_mean = float(np.mean(neg_sims))

        # optional: similarity to own prototype (stability term)
        pos_sim = 0.0
        if agent_id in self.protos:
            pos_sim = float(np.dot(emb, self.protos[agent_id]))

        return pos_sim - neg_mean
    
    def cosine_matrix(self, agent_ids=None):
        if agent_ids is None:
            agent_ids = sorted(self.protos.keys())

        if len(agent_ids) == 0:
            return agent_ids, None

        # stack prototypes in consistent order
        P = np.stack([self.protos[i] for i in agent_ids], axis=0)  # (n, dim)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        S = P @ P.T  # cosine similarity matrix
        return agent_ids, S

    def log_episode(self, agent_id, emb, max_per_agent=5000):
        emb = self._norm(emb)
        if agent_id not in self.episode_embs:
            self.episode_embs[agent_id] = []
        if len(self.episode_embs[agent_id]) < max_per_agent:
            self.episode_embs[agent_id].append(emb.copy())

    def init_distance_logger(self, path="proto_logs/prototype_cosine_over_time.txt", agent_ids=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.log_path = path
        self._log_t0 = time.time()
        self._log_step = 0
        self._log_agent_ids = agent_ids  # optional fixed order

        with open(self.log_path, "w") as f:
            f.write("# step, elapsed_sec, agent_ids, cosine_matrix_flat_rowmajor\n")

    def _get_agent_ids_for_log(self):
        if getattr(self, "_log_agent_ids", None) is not None:
            return list(self._log_agent_ids)
        return sorted(self.protos.keys())

    def log_cosine_snapshot(self):
        # log only when we have >= 2 agents with prototypes
        if len(self.protos) < 2:
            return

        agent_ids = self._get_agent_ids_for_log()
        # keep only those that exist so it never crashes
        agent_ids = [aid for aid in agent_ids if aid in self.protos]
        if len(agent_ids) < 2:
            return

        # stack + normalize
        P = np.stack([self.protos[aid] for aid in agent_ids], axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        S = P @ P.T  # cosine similarity matrix

        self._log_step += 1
        elapsed = time.time() - self._log_t0

        # flatten row-major so it fits on one line
        S_flat = S.reshape(-1)

        with open(self.log_path, "a") as f:
            f.write(f"{self._log_step},{elapsed:.6f},{agent_ids},{S_flat.tolist()}\n")



class NetworkAttackEnv(gym.Env):
    """
    Simple RL environment for an adversarial perturber (Red Agent).
    The agent applies perturbations to a non-benign sample to fool a classifier into classifying it as benign.
    """

    def __init__(self, classifier, X_data, y_data, benign_label=0, max_steps=25, epsilon=0.25, agent_id=0, contrastive_bank=None, alpha_contrast=0.5, target_margin_confidence=None, shift_reference_classifier=None, pocket_shift_weight=None, proximity_anchor_X=None, proximity_weight=None, proximity_length_scale=5.0):
        super(NetworkAttackEnv, self).__init__()
        print(f"\n==== Agent Epsilon is -{epsilon} and {epsilon} ===== ")

        self.classifier = classifier  # Pre-trained classifier
        self.X_data = X_data  # Input samples
        self.y_data = y_data  # True labels
        self.benign_label = benign_label  # Target benign class
        self.max_steps = max_steps  # Max perturbation attempts per episode
        self.epsilon = epsilon  # Perturbation strength
        self.last_sample_index = 0
        # None (default) = today's behavior: monotonically reward pushing
        # confidence as deep into the target class as possible. When set
        # (e.g. 0.55), reward instead PEAKS at that confidence value and
        # falls off in both directions -- so the agent is rewarded for
        # landing just past the decision boundary rather than deep past it.
        # See step()'s confidence_term for the exact shape.
        self.target_margin_confidence = target_margin_confidence
        # REWORK (pocket-targeted test poisoning): shift_reference_classifier,
        # when set, is a FROZEN snapshot of `classifier` from an earlier point
        # in time (e.g. before this task's CL training ran) -- `classifier`
        # itself is expected to be the CURRENT/live model. pocket_shift_weight
        # scales a reward bonus for the SIGNED CHANGE in confidence toward
        # `benign_label` between the two, at the agent's current perturbed
        # point: positive when the live model classifies this exact point
        # MORE toward benign_label than the frozen reference did. This
        # specifically hunts for "pockets" the boundary has moved through --
        # points a prior classifier state would have called non-benign that
        # the current one now calls benign -- rather than just any generic
        # evasion direction. See step() for the exact reward term. Both must
        # be set together for this to activate; either left None (default)
        # disables it and reward is unchanged from before.
        self.shift_reference_classifier = shift_reference_classifier
        self.pocket_shift_weight = pocket_shift_weight
        # REWORK (proximity-anchored pocket targeting): proximity_anchor_X, when
        # set, is a small array of raw ([0,1]) feature vectors -- typically THIS
        # task's actual TRAIN-side poisoned exemplars, the exact rows that will
        # become the forget set. proximity_weight scales a reward bonus that
        # peaks (at proximity_weight) when the agent's current point sits AT an
        # anchor and decays smoothly (exp(-nearest_dist / proximity_length_scale))
        # as it moves away -- pulling this INDEPENDENTLY-trained agent toward the
        # same neighborhood unlearning will actually correct, without forcing it
        # to reproduce the exact same perturbation (still a genuinely separate,
        # RL-discovered attack, still testing real generalization). Both
        # proximity_anchor_X and proximity_weight must be set together for this
        # to activate. proximity_length_scale is a first-pass guess (raw feature
        # space is unnormalized here, so this should be tuned against actually
        # observed anchor distances for a given dataset).
        self.proximity_anchor_X = (
            np.asarray(proximity_anchor_X, dtype=np.float32)
            if proximity_anchor_X is not None and len(proximity_anchor_X) > 0 else None
        )
        self.proximity_weight = proximity_weight
        self.proximity_length_scale = proximity_length_scale

        self.non_benign_indices = np.where(y_data != benign_label)[0]
        feature_dim = X_data.shape[1]  # Ensure action space matches feature dimension

        self.observation_space = spaces.Box(low=0, high=1, shape=(feature_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-epsilon, high=epsilon, shape=(feature_dim,), dtype=np.float32)

        self.current_index = 0
        self.steps = 0
        self.state = None
        self.true_label = None
        self.rewards_list = []

        self.agent_id = agent_id
        self.contrastive_bank = contrastive_bank
        self.alpha_contrast = alpha_contrast
        self.base_state = None
        self.cum_action = None

        self.reset()



        

    def update_data(self, x_data,y_data,benign_label ): #to update used samples
        self.X_data = x_data
        self.y_data = y_data
        self.non_benign_indices = np.where(y_data != benign_label)[0]

    def step(self, action):
        # print("THUS US THE ACTION BEFORE", action)
        # action = np.clip(action, -self.epsilon, self.epsilon)

        # Adding pertrubation to the network packet (state)
        self.state = np.clip(self.state + action, 0, 1)

        # Getting the cummulative pertubations. 
        self.cum_action += action.astype(np.float32)

        predicted_proba = self.classifier.predict_proba(self.state.reshape(1, -1))[0]
        predicted_label = np.argmax(predicted_proba)

        r_contrast = 0.0
        if self.contrastive_bank is not None:
            emb = self.cum_action  # behavior signature
            r_contrast = self.alpha_contrast * self.contrastive_bank.contrastive_reward(self.agent_id, emb)


        # Reward is based on classifier's confidence change
        confidence = predicted_proba[self.benign_label]  # Probability of benign class
        #reward = confidence - 0.5  # Encourage increasing benign probability
        if self.target_margin_confidence is not None:
            # Margin-minimizing mode: peak reward (2.5, matching the default
            # mode's max at confidence=1.0) sits at target_margin_confidence
            # -- a small margin PAST 0.5 toward the target class -- and falls
            # off symmetrically as confidence moves either further short of
            # or further past that point, unlike the default's monotonic
            # "deeper is always better".
            confidence_term = 5 * (0.5 - abs(confidence - self.target_margin_confidence))
        else:
            confidence_term = 5 * (confidence - 0.5)

        # REWORK (pocket-targeted test poisoning): bonus for landing where the
        # CURRENT (live) classifier's confidence toward benign_label at this
        # exact point exceeds the FROZEN reference classifier's confidence at
        # the same point -- i.e. this specific point is inside a region the
        # boundary moved through, not just generically evasive. Zero/no-op
        # unless both shift_reference_classifier and pocket_shift_weight are
        # set (see __init__).
        pocket_term = 0.0
        if self.shift_reference_classifier is not None and self.pocket_shift_weight:
            ref_proba = self.shift_reference_classifier.predict_proba(self.state.reshape(1, -1))[0]
            ref_confidence = ref_proba[self.benign_label]
            pocket_term = self.pocket_shift_weight * (confidence - ref_confidence)

        # REWORK (proximity-anchored pocket targeting): bonus that peaks when
        # the agent's current point sits at (or very near) one of
        # proximity_anchor_X's rows, decaying smoothly with distance -- pulls
        # this attack toward the SAME region unlearning will forget. Zero/no-op
        # unless both proximity_anchor_X and proximity_weight are set.
        proximity_term = 0.0
        if self.proximity_anchor_X is not None and self.proximity_weight:
            dists = np.linalg.norm(self.proximity_anchor_X - self.state.reshape(1, -1), axis=1)
            nearest_dist = float(dists.min())
            proximity_term = self.proximity_weight * np.exp(-nearest_dist / self.proximity_length_scale)

        reward = confidence_term + pocket_term + proximity_term - 0.3 * np.linalg.norm(action, ord=2)  # L2 penalty

        # Large reward for successful misclassification
        if predicted_label == self.benign_label:
            reward += 10

        reward = reward + r_contrast
        # print("Agent ID: ", self.agent_id, "\nContrast Reward: ", r_contrast)


        # self.steps += 1
        # done = (self.steps >= self.max_steps) or (predicted_label == self.benign_label)

        # if done and self.contrastive_bank is not None:
        #     self.contrastive_bank.update(self.agent_id, self.cum_action)

        # self.rewards_list.append(reward)

        # if done and self.contrastive_bank is not None:
        #     self.contrastive_bank.update(self.agent_id, self.cum_action)
        #     self.contrastive_bank.log_episode(self.agent_id, self.cum_action)

        # return self.state, reward, done, {"prediction":predicted_label}

        self.steps += 1
        terminated = bool(predicted_label == self.benign_label)
        truncated = bool(self.steps >= self.max_steps)

        if (terminated or truncated) and self.contrastive_bank is not None:
            self.contrastive_bank.update(self.agent_id, self.cum_action)
            self.contrastive_bank.log_episode(self.agent_id, self.cum_action)
            self.contrastive_bank.log_cosine_snapshot()


        reward = float(reward)
        info = {"prediction": int(predicted_label)}

        return self.state, reward, terminated, truncated, info


       

    # def reset(self, index=0):
    #     print("Hit the correct reset\n\n\n")
    #     if len(self.non_benign_indices) == 0:
    #         raise ValueError("\nOnly benign samples found! Check dataset labels.")
    #     '''
    #     self.current_index = np.random.choice(self.non_benign_indices)
    #     self.state = self.X_data[self.current_index].astype(np.float32).copy()
    #     self.true_label = self.y_data[self.current_index]
    #     self.steps = 0
    #     '''
    #     if index == 0:
    #         self.current_index = np.random.choice(self.non_benign_indices)
    #     elif index in self.non_benign_indices:
    #         self.current_index = index
    #     else:
    #         self.current_index = np.random.choice(self.non_benign_indices)
    #     self.state = np.array(self.X_data[self.current_index], dtype=np.float32).copy()

    #     self.base_state = self.state.copy()
    #     self.cum_action = np.zeros_like(self.state, dtype=np.float32)

    #     self.true_label = self.y_data[self.current_index]
    #     self.steps = 0

    #     self.last_sample_index = self.current_index

    #     #print(f"New episode started: Perturbing sample {self.current_index}, Original Label: {self.true_label}, Benign Label: {self.benign_label}")
        
    #     avg = np.mean(self.rewards_list)
    #     std_dev = np.std(self.rewards_list) 
    #     with open("cumulative_r_training_red.txt", "a") as f:  # 'a' mode appends instead of overwriting
    #         f.write(f"{avg}\n")
    #     with open("stand_dev_r_training_red.txt", "a") as g: 
    #         g.write(f"{std_dev}\n")

    #     self.rewards_list = []
    #     return self.state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if len(self.non_benign_indices) == 0:
            raise ValueError("\nOnly benign samples found! Check dataset labels.")

        index = 0
        if options is not None and "index" in options:
            index = options["index"]

        if index == 0:
            self.current_index = np.random.choice(self.non_benign_indices)
        elif index in self.non_benign_indices:
            self.current_index = index
        else:
            self.current_index = np.random.choice(self.non_benign_indices)

        self.state = np.array(self.X_data[self.current_index], dtype=np.float32).copy()
        self.base_state = self.state.copy()
        self.cum_action = np.zeros_like(self.state, dtype=np.float32)
        self.true_label = self.y_data[self.current_index]
        self.steps = 0
        self.last_sample_index = self.current_index

        # your logging (optional)
        avg = np.mean(self.rewards_list) if self.rewards_list else 0.0
        std_dev = np.std(self.rewards_list) if self.rewards_list else 0.0
        # with open("cumulative_r_training_red.txt", "a") as f:
        #     f.write(f"{avg}\n")
        # with open("stand_dev_r_training_red.txt", "a") as g:
        #     g.write(f"{std_dev}\n")
        self.rewards_list = []

        info = {"index": int(self.current_index), "true_label": int(self.true_label)}
        return self.state, info


    def render(self, mode='human'):
        pass  # Not needed for this simple setup

    def seed(self, seed=None):
        np.random.seed(seed)
