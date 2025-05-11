import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    # NEW:
    def __init__(self):
        self.reward_type = "individual" # default, will be overriden by MPEEnv()
    # NEW END
    
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        world.prev_total_distance = None
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    # OLD:
    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
    #                  for a in world.agents]
    #         rew -= min(dists)

    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #     return rew

    # NEW:
    def reward(self, agent, world):
        """Implements individual, shared, or partially shared reward structure."""
        def individual():
            ideal_dist = 0.25
            tolerance = 0.1
            bonus = 1.0
            penalty_close = -1.0
            penalty_far = -0.5
            collision_penalty = -1.0

            # Find nearest landmark
            distances = [np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks]
            min_dist = min(distances)

            # Reward shaping
            if abs(min_dist - ideal_dist) < tolerance:
                proximity_reward = bonus
            elif min_dist < ideal_dist:
                proximity_reward = penalty_close
            else:
                proximity_reward = penalty_far

            # Penalize collisions
            collision_loss = 0
            if agent.collide:
                for other in world.agents:
                    if other is not agent and self.is_collision(agent, other):
                        collision_loss += collision_penalty

            return proximity_reward + collision_loss

        def shared():
            total_dist = 0
            for l in world.landmarks:
                for a in world.agents:
                    total_dist += np.linalg.norm(a.state.p_pos - l.state.p_pos)

            if not hasattr(world, "prev_total_dist") or world.prev_total_dist is None:
                world.prev_total_dist = total_dist

            progress = world.prev_total_dist - total_dist
            world.prev_total_dist = total_dist

            # Penalize collisions
            collision_penalty = 0
            for i, a1 in enumerate(world.agents):
                for j, a2 in enumerate(world.agents):
                    if i != j and self.is_collision(a1, a2):
                        collision_penalty -= 1

            return progress + collision_penalty


        def partially_shared():
            # Individual component
            individual_rewards = []
            for agent in world.agents:
                dists = [np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks]
                min_dist = min(dists)
                individual_rewards.append(-min_dist)

            # Shared component (mean distance to landmarks)
            total_dist = 0
            for l in world.landmarks:
                for a in world.agents:
                    total_dist += np.linalg.norm(a.state.p_pos - l.state.p_pos)

            if not hasattr(world, "prev_total_dist") or world.prev_total_dist is None:
                world.prev_total_dist = total_dist

            shared_progress = world.prev_total_dist - total_dist
            world.prev_total_dist = total_dist

            # Coverage bonus: reward if each landmark is closest to a different agent
            closest_agents = [np.argmin([np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]) for l in world.landmarks]
            coverage_bonus = len(set(closest_agents))  # Higher if agents spread out

            return 0.6 * individual_rewards[world.agents.index(agent)] + 0.4 * (shared_progress + coverage_bonus * 0.1)

        if self.reward_type == "individual":
            return individual()
        elif self.reward_type == "shared":
            return shared()
        elif self.reward_type == "partially_shared":
            # Blend individual and shared rewards
            # 0.5 * individual + 0.5 * shared
            return partially_shared()
        elif self.reward_type == "original":
            rew = 0
            for l in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                         for a in world.agents]
                rew -= min(dists)
                
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
            return rew
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")   
    # NEW END

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
