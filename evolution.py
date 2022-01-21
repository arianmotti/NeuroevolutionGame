import copy
import math
import random
import numpy as np

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.rr

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players.sort(key=lambda x: x.fitness, reverse=True)
        # print(len(players),num_players)

        # TODO (Additional: Implement roulette wheel here)
        players2 = self.roulette_wheel(players, num_players)

        # TODO (Additional: Implement SUS here)
        players3 = self.sus(players, num_players)

        # TODO (Additional: Learning curve)
        players4 = self.learning_curve(players, num_players)

        print(sum(player.fitness for player in players[: num_players]) / num_players)
        print(players[0].fitness)
        print(players[num_players - 1].fitness)
        self.plot_data_saving(players[: num_players])

        return players[: num_players]

    def roulette_wheel(self, players, num_players):
        players2 = []
        max = sum(player.fitness for player in players)
        for i in range(num_players):
            pick = random.uniform(0, max)
            current = 0
            for player in players:
                current += player.fitness
                if current > pick:
                    players2.append(player)
                    break
        return players2

    def sus(self, players, num_players):
        sum_of_fitness = np.sum([x.fitness for x in players])
        step_size = sum_of_fitness / num_players

        # creating the ruler
        ruler = np.arange(num_players) * step_size
        random_number = np.random.uniform(0, step_size)
        ruler = ruler + random_number

        selected_players = []
        for r in ruler:
            i = 0
            f = 0
            while f < r:
                f += players[i].fitness
                i += 1
            selected_players.append(players[i - 1])
        return selected_players

    def learning_curve(self, players, num_players):
        # q = 3
        players4 = []
        for i in range(num_players):

            tmp1 = players[random.randint(0, len(players) - 1)]
            max = tmp1
            tmp2 = players[random.randint(0, len(players) - 1)]
            if tmp2.fitness > max.fitness:
                max = tmp2
            tmp3 = players[random.randint(0, len(players) - 1)]
            if tmp3.fitness > max.fitness:
                max = tmp3
            players4.append(max)
        return players4

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            prev_players=self.roulette_wheel(prev_players,num_players)
            new_players = self.cross_over(prev_players, num_players)
            children = [self.mutate(x, 0.2, 0.4, 0.2) for x in new_players]

            return children

    def mutate(self, child, pw, pb, d):

        # child: an object of class `Player`
        child = self.clone_player(child)
        mutation_prob = np.random.uniform(0, 1)
        if mutation_prob <= pw:
            child.nn.w1 += np.random.normal(0, d, child.nn.w1.shape)
        mutation_prob = np.random.uniform(0, 1)
        if mutation_prob <= pb:
            child.nn.b1 += np.random.normal(0, d, child.nn.b1.shape)
        mutation_prob = np.random.uniform(0, 1)
        if mutation_prob <= pw:
            child.nn.w2 += np.random.normal(0, d, child.nn.w2.shape)
        mutation_prob = np.random.uniform(0, 1)
        if mutation_prob <= pb:
            child.nn.b2 += np.random.normal(0, d, child.nn.b2.shape)

        return child

    def cross_over(self, players, num_players):
        children = []
        index = 0
        for i in range(math.floor(num_players / 2.0)):
            cross_over_prob = np.random.uniform(0, 1)
            p = 0.8
            if cross_over_prob >= p:
                children.append(players[index])
                children.append(players[index + 1])
                index += 2
                continue
            parent1 = players[index]
            parent2 = players[index + 1]
            child1 = self.clone_player(parent1)
            child2 = self.clone_player(parent2)

            d0 = math.floor(parent1.nn.sizes[0] / 2)
            d1 = math.floor(parent1.nn.sizes[1] / 2)
            d2 = math.floor(parent1.nn.sizes[2] / 2)
            child1.nn.w1 = np.concatenate((parent1.nn.w1[:d1], parent2.nn.w1[d1:]), axis=0)
            child1.nn.b1 = np.concatenate((parent1.nn.b1[:d1], parent2.nn.b1[d1:]), axis=0)
            child1.nn.w2 = np.concatenate((parent1.nn.w2[:d2], parent2.nn.w2[d2:]), axis=0)
            child1.nn.b2 = np.concatenate((parent1.nn.b2[:d2], parent2.nn.b2[d2:]), axis=0)
            children.append(child1)
            child2.nn.w1 = np.concatenate((parent2.nn.w1[:d1], parent1.nn.w1[d1:]), axis=0)
            child2.nn.b1 = np.concatenate((parent2.nn.b1[:d1], parent1.nn.b1[d1:]), axis=0)
            child2.nn.w2 = np.concatenate((parent2.nn.w2[:d2], parent1.nn.w2[d2:]), axis=0)
            child2.nn.b2 = np.concatenate((parent2.nn.b2[:d2], parent1.nn.b2[d2:]), axis=0)
            children.append(child2)
            index += 2
        if len(children) < num_players:
            children.append(players[0])

        return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def plot_data_saving(self, players):
        with open(f'data_files/max.csv', 'a') as file:
            file.write(str(sorted(players, key=lambda x: x.fitness, reverse=True)[0].fitness))
            file.write("\n")
        with open(f'data_files/min.csv', 'a') as file:
            file.write(str(sorted(players, key=lambda x: x.fitness)[0].fitness))
            file.write("\n")
        ave = 0
        for p in players:
            ave += p.fitness
        ave /= len(players)
        with open(f'data_files/average.csv', 'a') as file:
            file.write(str(ave))
            file.write("\n")
