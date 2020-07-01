//
// Created by Yusuke Arai on 2020/05/08.
//

#ifndef HNSW_HNSW_HPP
#define HNSW_HNSW_HPP

#include <cmath>
#include <random>
#include <map>
#include <arailib.hpp>

using namespace std;
using namespace arailib;

namespace hnsw {
    struct Node {
        const Data<>& data;
        vector<int> neighbors;

        explicit Node(const Data<>& data_) : data(data_) {}
    };

    using Layer = vector<Node>;

    struct SearchResult {
        vector<int> result;
        time_t time = 0;
        unsigned int n_dist_calc = 0;
        unsigned int n_dist_calc_upper_layer = 0;
        unsigned int n_dist_calc_base_layer = 0;
        unsigned int n_hop = 0;
        unsigned int n_hop_upper_layer = 0;
        unsigned int n_hop_base_layer = 0;
        double dist_from_ep_base_layer = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;

        SearchResults(size_t size) : results(size) {}
        void push_back(const SearchResult& result) { results.emplace_back(result); }
        void push_back(SearchResult&& result) { results.emplace_back(move(result)); }
        decltype(auto) operator [] (int i) { return results[i]; }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "time,n_dist_calc,n_dist_calc_upper_layer,n_dist_calc_base_layer,"
                          "n_hop,n_hop_upper_layer,n_hop_base_layer,dist_from_ep_base_layer";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                log_ofs << result.time << ","
                        << result.n_dist_calc << ","
                        << result.n_dist_calc_upper_layer << ","
                        << result.n_dist_calc_base_layer << ","
                        << result.n_hop << ","
                        << result.n_hop_upper_layer << ","
                        << result.n_hop_base_layer << ","
                        << result.dist_from_ep_base_layer << endl;

                for (const auto& data_id : result.result) {
                    result_ofs << query_id << ","
                               << data_id << endl;
                }

                query_id++;
            }
        }
    };

    unsigned int in_dist_calc_counter = 0;
    unsigned int in_hop_counter = 0;

    struct HNSW {
        const int m, m_max_0;
        const double m_l;
        const bool extend_candidates, keep_pruned_connections;

        int enter_node_id;
        vector<Layer> layers;
        map<int, vector<int>> layer_map;
        Dataset<> dataset;

        mt19937 engine;
        uniform_real_distribution<double> unif_dist;

        HNSW(int m, bool extend_candidates = true, bool keep_pruned_connections = true) :
                m(m), m_max_0(m * 2), m_l(1 / log(1.0 * m)),
                extend_candidates(extend_candidates),
                keep_pruned_connections(keep_pruned_connections),
                engine(42), unif_dist(0.0, 1.0) {}

        auto get_max_layer() const { return layers.size() - 1; }

        const Node& get_enter_node() const { return layers.back()[enter_node_id]; }

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        auto search_layer(const Data<>& query, int start_node_id, int ef, int l_c,
                          unsigned int& dist_calc_counter = in_dist_calc_counter,
                          unsigned int& hop_counter = in_hop_counter) {
            unordered_map<int, bool> visited;
            visited[start_node_id] = true;

            const auto& start_node = layers[l_c][start_node_id];
            const auto dist_from_en = euclidean_distance(query, start_node.data);

            map<double, int> candidate_map;
            candidate_map.emplace(dist_from_en, start_node_id);

            map<double, int> result_map;
            result_map.emplace(dist_from_en, start_node_id);

            while (!candidate_map.empty()) {
                const auto nearest_candidate_itr = candidate_map.cbegin();
                const auto dist_from_nearest_candidate = nearest_candidate_itr->first;
                const auto nearest_candidate_id = nearest_candidate_itr->second;
                const auto& nearest_candidate = layers[l_c][nearest_candidate_id];
                candidate_map.erase(nearest_candidate_itr);

                const auto furthest_result_itr = --result_map.cend();
                const auto dist_from_furthest_result = furthest_result_itr->first;

                if (dist_from_nearest_candidate > dist_from_furthest_result) break;

                ++hop_counter;

                for (const auto neighbor_id : nearest_candidate.neighbors) {
                    if (visited[neighbor_id]) continue;
                    visited[neighbor_id] = true;

                    const auto& neighbor = layers[l_c][neighbor_id];
                    const auto dist_from_neighbor =
                            euclidean_distance(query, neighbor.data);
                    ++dist_calc_counter;
                    const auto furthest_result_itr_ = --result_map.cend();
                    const auto dist_from_furthest_result_ = furthest_result_itr_->first;

                    if (dist_from_neighbor < dist_from_furthest_result_ ||
                        result_map.size() < ef) {
                        candidate_map.emplace(dist_from_neighbor, neighbor_id);
                        result_map.emplace(dist_from_neighbor, neighbor_id);

                        if (result_map.size() > ef)
                            result_map.erase(furthest_result_itr_);
                    }
                }
            }

            vector<int> result;
            for (const auto& result_pair : result_map)
                result.emplace_back(result_pair.second);
            return result;
        }

        auto select_neighbors_heuristic(const Data<>& query, vector<int> initial_candidates,
                                        int n_neighbors, int l_c) {
            map<double, int> result_map;
            map<double, int> candidate_map;

            for (const auto& candidate_id : initial_candidates) {
                const auto& candidate = layers[l_c][candidate_id];
                const auto dist_from_candidate =
                        euclidean_distance(query, candidate.data);
                candidate_map.emplace(dist_from_candidate, candidate_id);
            }

            if (extend_candidates) {
                for (const auto& candidate_id : initial_candidates) {
                    // candidate must be get like below (not to get ref of layers[l_c])
                    const auto& candidate = layers[l_c][candidate_id];
                    for (const auto& neighbor_id : candidate.neighbors) {
                        const auto& neighbor = layers[l_c][neighbor_id];
                        const auto dist_from_neighbor =
                                euclidean_distance(query, neighbor.data);
                        candidate_map.emplace(dist_from_neighbor, neighbor_id);
                    }
                }
            }

            map<double, int> discarded_candidate_map;
            while (!candidate_map.empty() && result_map.size() < m) {
                const auto candidate_itr = candidate_map.cbegin();
                const auto dist_from_candidate = candidate_itr->first;
                const auto candidate_id = candidate_itr->second;
                candidate_map.erase(candidate_itr);

                if (candidate_id == query.id) continue;
                const auto& candidate = layers[l_c][candidate_id];

                if (result_map.empty()) {
                    result_map.emplace(dist_from_candidate, candidate_id);
                    continue;
                }

                const auto furthest_result_itr = --result_map.cend();
                const auto dist_from_furthest_result = furthest_result_itr->first;

                if (dist_from_candidate < dist_from_furthest_result)
                    result_map.emplace(dist_from_candidate, candidate_id);
                else
                    discarded_candidate_map.emplace(dist_from_candidate, candidate_id);
            }

            if (keep_pruned_connections) {
                while (!discarded_candidate_map.empty()
                       && result_map.size() < n_neighbors) {
                    const auto discarded_candidate_itr = discarded_candidate_map.cbegin();
                    const auto discarded_candidate_pair = *discarded_candidate_itr;
                    discarded_candidate_map.erase(discarded_candidate_itr);
                    result_map.emplace(discarded_candidate_pair);
                }
            }

            vector<int> result;
            for (const auto& result_pair : result_map)
                result.emplace_back(result_pair.second);
            return result;
        }

        void insert(const Data<>& new_data) {
            bool changed_enter_node = false;
            auto l_new_node = get_new_node_level();
            for (int l_c = l_new_node; l_c >= 0; --l_c)
                layer_map[l_c].emplace_back(new_data.id);

            if (layers.empty() || l_new_node > get_max_layer()) {
                // add new layer
                int l_c = get_max_layer() + 1;
                layers.resize(l_new_node + 1);
                for (; l_c <= l_new_node; ++l_c) {
                    for (const auto& data : dataset) {
                        layers[l_c].emplace_back(data);
                    }
                }
                changed_enter_node = true;
            }

            auto start_node_id = enter_node_id;
            for (int l_c = get_max_layer(); l_c > l_new_node; --l_c) {
                const auto nn_id_layer = search_layer(new_data, start_node_id, 1, l_c)[0];
                start_node_id = nn_id_layer;
            }

            for (int l_c = l_new_node; l_c >= 0; --l_c) {
                const auto neighbors = search_layer(new_data, start_node_id, m, l_c);
                auto& layer = layers[l_c];
                for (const auto neighbor_id : neighbors) {
                    if (neighbor_id == new_data.id) continue;
                    auto& neighbor = layer[neighbor_id];

                    // add a bidirectional edge
                    layer[new_data.id].neighbors.emplace_back(neighbor_id);
                    neighbor.neighbors.emplace_back(new_data.id);

                    const auto m_max = l_c == 0 ? m_max_0 : m;

                    if (neighbor.neighbors.size() > m_max) {
                        auto new_neighbor_neighbors = select_neighbors_heuristic(
                                neighbor.data, neighbor.neighbors, m_max + 1, l_c);
                        neighbor.neighbors = new_neighbor_neighbors;
                    }
                }
                if (l_c == 0) break;
                start_node_id = neighbors[0];
            }

            if (changed_enter_node)
                enter_node_id = new_data.id;
        }

        void build(const Dataset<>& dataset_) {
            dataset = dataset_;
            for (const auto& data : dataset) insert(data);
        }

        SearchResult knn_search(const Data<>& query, int k, int ef) {
            SearchResult result;
            const auto begin = get_now();

            auto start_id_layer = enter_node_id;
            for (int l_c = get_max_layer(); l_c >= 1; --l_c) {
                const auto nn_id_layer = search_layer(query, start_id_layer, 1, l_c,
                                                      result.n_dist_calc_upper_layer,
                                                      result.n_hop_upper_layer)[0];
                start_id_layer = nn_id_layer;
            }

            const auto& nn_upper_layer = layers[1][start_id_layer];

            const auto candidates = search_layer(query, start_id_layer, ef, 0,
                                                 result.n_dist_calc_base_layer,
                                                 result.n_hop_base_layer);
            for (const auto& candidate_id : candidates) {
                const auto& candidate = dataset[candidate_id];
                result.result.emplace_back(candidate_id);
                if (result.result.size() >= k) break;
            }

            const auto end = get_now();
            result.time = get_duration(begin, end);
            result.n_dist_calc = result.n_dist_calc_upper_layer + result.n_dist_calc_base_layer;
            result.n_hop = result.n_hop_upper_layer + result.n_hop_base_layer;
            result.dist_from_ep_base_layer = euclidean_distance(query, nn_upper_layer.data);

            return result;
        }
    };
}

#endif //HNSW_HNSW_HPP
