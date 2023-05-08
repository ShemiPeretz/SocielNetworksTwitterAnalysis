import json
import pickle
import re
from datetime import timedelta, datetime
import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.community import girvan_newman, k_clique_communities
import gzip


# Q1


def community_detector(algorithm_name: str, network: nx.Graph, most_valuable_edge=None) -> dict:
    if algorithm_name == 'girvan_newman':
        if most_valuable_edge:
            communities = list(girvan_newman(network, most_valuable_edge=most_valuable_edge))
        else:
            communities = list(girvan_newman(network))
        maxModularity = 0
        for option in communities:
            modularity = nx.algorithms.community.modularity(network, option)
            if modularity >= maxModularity:
                maxModularity = modularity
                partition = option
                num_partitions = len(partition)
        modularity = maxModularity
    elif algorithm_name == 'louvain':
        communities = nx.community.louvain_communities(network)
        partition = communities
        modularity = nx.algorithms.community.modularity(network, communities)
        num_partitions = len(partition)
    elif algorithm_name == 'clique_percolation':
        # Get the communities
        k = 3  # k-clique value, can be modified
        communities = list(k_clique_communities(network, k))
        # Handle overlapping nodes in the communities for the modularity algorithm
        nonOverlappingCommunities = [list(communities[0])]
        for i in range(1, len(communities)):
            community = communities[i]
            tmp = []
            for node in community:
                if node not in [n for c in nonOverlappingCommunities for n in c]:
                    tmp.append(node)
            nonOverlappingCommunities.append(tmp)
        # Handle missing nodes in the partition
        communities_ = [n for c in nonOverlappingCommunities for n in c]
        for node in G.nodes:
            if node not in communities_:
                nonOverlappingCommunities.append([node])
        # Calc results
        partition = nonOverlappingCommunities
        modularity = nx.algorithms.community.modularity(network, nonOverlappingCommunities)
        num_partitions = len(partition)
    else:
        raise ValueError(
            "Invalid algorithm name. Supported algorithms are 'girvan_newman', 'louvain' and 'clique_percolation'.")

    result_dict = {
        'num_partitions': num_partitions,
        'modularity': modularity,
        'partition': partition
    }

    return result_dict


def edge_selector_optimizer(network):
    betweenness = nx.edge_betweenness_centrality(network)
    return max(betweenness, key=betweenness.get)


# Load the Les Miserables network
G = nx.les_miserables_graph()


## Q1 - code for iii

# # Run the community detection function with different algorithms
# gn_result = community_detector('girvan_newman', G)
# gn_eso_result = community_detector('girvan_newman', G, edge_selector_optimizer)
# lv_result = community_detector('louvain', G)
# cp_result = community_detector('clique_percolation', G)
# all_results = {'girvan_newman': gn_result, "Girvan-Newman with Edge selector" :gn_eso_result,
#                "Louvain": lv_result, "Clique percolation": cp_result}
# # Some global statistics
# best_modularity = 0
# for algorithm, result in all_results.items():
#     if result["modularity"] >= best_modularity:
#         best_modularity = result["modularity"]
#         best_modularity_algorithm = algorithm
#
# # Print the results
# print("Girvan-Newman algorithm:")
# print(gn_result)
# print()
# print("Girvan-Newman with Edge selector optimizer algorithm:")
# print(gn_eso_result)
# print()
# print("Louvain algorithm:")
# print(lv_result)
# print()
# print("Clique percolation algorithm:")
# print(cp_result)
# print()
# print(f"Best modularity = {best_modularity}, yielded from {best_modularity_algorithm} algorithm")


# Q2


def construct_heb_edges(files_path: str, start_date: str = '2019-03-15', end_date: str = '2019-04-15',
                        non_parliamentarians_nodes: int = 0) -> dict:
    centralPoliticalPlayers = pd.read_csv(f"{files_path}\\central_political_players_2019.csv")
    allowedDates = [str(dateTime).split(" ")[0] for dateTime in pd.date_range(start=start_date, end=end_date,
                                                                              inclusive="both", freq='D')]
    rawTweetsFiles = [open(f"{files_path}\\Hebrew_twitter_data_2019\\Hebrew_tweets.json.{date}.txt") for date in allowedDates]

    retweets_dict = {}
    non_parliamentarians_nodes_retweets_dict = {}
    influence_of_nodes = {}
    for file in rawTweetsFiles:
        for line in file.readlines():
            jsonTweet = json.loads(line)
            if 'retweeted_status' in jsonTweet and jsonTweet['retweeted_status'] != "":
                user_id = jsonTweet['user']['id']
                retweeted_user_id = jsonTweet['retweeted_status']['user']['id']
                if user_id not in centralPoliticalPlayers.id.values or retweeted_user_id not in centralPoliticalPlayers.id.values:
                    if user_id not in centralPoliticalPlayers.id.values and retweeted_user_id not in centralPoliticalPlayers.id.values:
                        # Choosing only retweets of central political nodes
                        continue
                    # One of users is a non_parliamentarian_nodes
                    key_tup = (user_id, retweeted_user_id)
                    non_parliamentarians_nodes_retweets_dict.setdefault(key_tup, 0)
                    non_parliamentarians_nodes_retweets_dict[key_tup] += 1
                    # Adding relevant information for measuring the influence of a node
                    if 'reply_count' in jsonTweet and 'retweet_count' in jsonTweet and 'favorite_count' in jsonTweet:
                        # get statistics only on well informative tweets
                        # count the replies, retweets and likes for each relevant non parliamentarian node
                        influence_of_nodes.setdefault(user_id,
                                                      {"retweeted_by_central": 0, "num_of_tweets": 0, "replies": 0,
                                                       "retweets": 0, "likes": 0})
                        influence_of_nodes[user_id]['num_of_tweets'] += 1
                        influence_of_nodes[user_id]['replies'] += jsonTweet['reply_count']
                        influence_of_nodes[user_id]['retweets'] += jsonTweet['retweet_count']
                        influence_of_nodes[user_id]['likes'] += jsonTweet['favorite_count']
                        if user_id in centralPoliticalPlayers.id.values:
                            # A central node retweeted a non-central
                            influence_of_nodes[user_id]['retweeted_by_central'] += 1
                else:  # both are central political nodes -> add to dict
                    key_tup = (user_id, retweeted_user_id)
                    retweets_dict.setdefault(key_tup, 0)
                    retweets_dict[key_tup] += 1

    influencing_nodes = get_non_central_influancers(non_parliamentarians_nodes, influence_of_nodes)
    for (retweeter, original_user) in non_parliamentarians_nodes_retweets_dict.keys():
        if retweeter in influencing_nodes:
            retweets_dict.setdefault((retweeter, original_user), 0)
            retweets_dict[(retweeter, original_user)] += 1

    return retweets_dict


def get_non_central_influancers(no_of_nodes: int, influence_of_nodes: dict):
    scores = {}
    for user, stats in influence_of_nodes.items():
        score = 0.6 * stats["retweeted_by_central"] + 0.1 * stats["num_of_tweets"] + 0.1 * stats["replies"] + 0.1 * stats["retweets"] + 0.1 * stats["likes"]
        scores[user] = score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_users = sorted_scores[:no_of_nodes]
    return top_users


def construct_heb_network(edge_dictionary: dict):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Iterate through the items in the dictionary and add edges to the graph
    for (retweeter, original_user), retweets_count in edge_dictionary.items():
        G.add_edge(retweeter, original_user, weight=retweets_count)
    return G


# Q3

def construct_heb_edges_2022(files_path: str, start_date: str = '2022-10-01', end_date: str = '2022-10-31',
                             non_parliamentarians_nodes: int = 0) -> dict:
    centralPoliticalPlayers = pd.read_csv(f"{files_path}\\central_political_players_2022.csv")
    allowedDates = [str(dateTime).split(" ")[0] for dateTime in pd.date_range(start=start_date, end=end_date,
                                                                              inclusive="both", freq='D')]
    allowedFileDatesNames = [f"Hebrew_tweets.json.{date}.gz" for date in allowedDates]
    retweets_dict = {}
    non_parliamentarians_nodes_retweets_dict = {}
    influence_of_nodes = {}

    for fileName in allowedFileDatesNames:
        path = f'{files_path}\\hebrew_twitter_data_2022_extractred\\{fileName}'
        counter = 0
        try:
            with gzip.open(path, 'r') as f:
                for line in f.readlines():
                    jsonTweet = json.loads(line)
                    counter += 1
                    if 'retweeted_status' in jsonTweet and jsonTweet['retweeted_status'] != "":
                        user_id = jsonTweet['user']['id']
                        retweeted_user_id = jsonTweet['retweeted_status']['user']['id']
                        if user_id not in centralPoliticalPlayers.id.values or retweeted_user_id not in centralPoliticalPlayers.id.values:
                            if user_id not in centralPoliticalPlayers.id.values and retweeted_user_id not in centralPoliticalPlayers.id.values:
                                # Choosing only retweets of central political nodes
                                continue
                            # One of users is a non_parliamentarian_nodes
                            key_tup = (user_id, retweeted_user_id)

                            non_parliamentarians_nodes_retweets_dict.setdefault(key_tup, 0)
                            non_parliamentarians_nodes_retweets_dict[key_tup] += 1
                            # Adding relevant information for measuring the influence of a node
                            if 'reply_count' in jsonTweet and 'retweet_count' in jsonTweet and 'favorite_count' in jsonTweet:
                                # get statistics only on well informative tweets
                                # count the replies, retweets and likes for each relevant non parliamentarian node
                                influence_of_nodes.setdefault(user_id,
                                                              {"retweeted_by_central": 0, "num_of_tweets": 0, "replies": 0,
                                                               "retweets": 0, "likes": 0})
                                influence_of_nodes[user_id]['num_of_tweets'] += 1
                                influence_of_nodes[user_id]['replies'] += jsonTweet['reply_count']
                                influence_of_nodes[user_id]['retweets'] += jsonTweet['retweet_count']
                                influence_of_nodes[user_id]['likes'] += jsonTweet['favorite_count']
                                if user_id in centralPoliticalPlayers.id.values:
                                    # A central node retweeted a non-central
                                    influence_of_nodes[user_id]['retweeted_by_central'] += 1
                        else:  # both are central political nodes -> add to dict
                            key_tup = (user_id, retweeted_user_id)
                            retweets_dict.setdefault(key_tup, 0)
                            retweets_dict[key_tup] += 1
        except:
            continue

    influencing_nodes = get_non_central_influancers(non_parliamentarians_nodes, influence_of_nodes)
    for (retweeter, original_user) in non_parliamentarians_nodes_retweets_dict.keys():
        if retweeter in influencing_nodes:
            retweets_dict.setdefault((retweeter, original_user), 0)
            retweets_dict[(retweeter, original_user)] += 1

    return retweets_dict

# Functions for plotting the networks


def get_nodes_weights(retweets_dictionary: dict, factor=1000):
    nodes = {}
    for key, val in retweets_dictionary.items():
        user2 = key[1]
        nodes.setdefault(user2, 0)
        nodes[user2] += val
    sum_of_weights = sum(nodes.values())
    for user1, user2 in retweets_dictionary.keys():
        if user1 not in nodes.keys():
            nodes.setdefault(user1, 1)
        else:
            nodes[user1] = (nodes[user1] / sum_of_weights) * factor

        if user2 not in nodes.keys():
            nodes.setdefault(user2, 1)
        else:
            nodes[user2] = (nodes[user2] / sum_of_weights) * factor
    return nodes


def get_nodes_weights_min_max_norm(retweets_dictionary: dict, factor=120):
    nodes = {}
    for key, val in retweets_dictionary.items():
        user1 = key[0]
        user2 = key[1]
        # {(1,2) : 10}
        nodes.setdefault(user2, 0)
        nodes[user2] += val
    max_val = max(nodes.values())
    min_val = min(nodes.values())
    for key, val in nodes.items():
        min_max_val = ((val - min_val) / (max_val - min_val))
        if min_max_val == 0:
            min_max_val = 10
        else:
            min_max_val *= factor
        nodes[key] = min_max_val
    return nodes

#  Code for Q2 & Q3

#
# files_path = "files"
# retweets_dict = construct_heb_edges(files_path)
# retweets_network = construct_heb_network(retweets_dict)
# retweets_dict_non_central = construct_heb_edges(files_path, non_parliamentarians_nodes=50)
# retweets_network_non_central = construct_heb_network(retweets_dict_non_central)
# retweets_dict_2022 = construct_heb_edges_2022(files_path)
# retweets_network_2022 = construct_heb_network(retweets_dict_2022)
# all_networks_dicts = [retweets_dict, retweets_dict_non_central, retweets_dict_2022]
# all_networks = [retweets_network, retweets_network_non_central, retweets_network_2022]
#
# centralPoliticalPlayers_2019 = pd.read_csv(f"{files_path}\\central_political_players_2022.csv")
# centralPoliticalPlayers_2022 = pd.read_csv(f"{files_path}\\central_political_players_2019.csv")
#
# # Applying girvin_newman to the network with no non_parliamentarians_nodes
# for i, network in enumerate(all_networks):
#     gn_retweets = community_detector('girvan_newman', retweets_network)
#     # Print the results
#     print("Girvan-Newman algorithm:")
#     print(gn_retweets)
#     print()
#
#     nt = Network('800px', width='1500px', directed=True)
#
#     colors = ["black", "brown", "gray", "red", "purple", "blue", "green",  "silver", "yellow", "pink", "cyan",
#               "orange", "olive", "dark blue", "coffee", "dark red"]
#     # nodes_weights = get_nodes_weights(retweets_dict)
#     nodes_weights = get_nodes_weights_min_max_norm(retweets_dict)
#     keys = retweets_dict.keys()
#
#     # populates the nodes and edges data structures
#     for i, c in enumerate(gn_retweets['partition']):
#         for node in c:
#             if i < 2:
#                 centralPoliticalPlayers = centralPoliticalPlayers_2019
#             else:
#                 centralPoliticalPlayers = centralPoliticalPlayers_2022
#             label = centralPoliticalPlayers.loc[centralPoliticalPlayers['id'] == node, 'name'].iloc[0]
#             if node in nodes_weights.keys():
#                 node_weight = nodes_weights[node]
#             else:
#                 node_weight = 10
#             if label:
#                 nt.add_node(node, label=label, color=colors[i], size=node_weight)
#             else:
#                 nt.add_node(node, color=colors[i], size=node_weight)
#
#     for node1, node2 in keys:
#         w = retweets_dict[(node1, node2)]
#         nt.add_edge(node1, node2, value=w, arrowStrikethrough=False)
#
#     nt.show_buttons()
#     nt.show(f'nx_{network.name}.html', notebook=False)

