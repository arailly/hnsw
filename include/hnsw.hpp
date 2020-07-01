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
        time_t time = 0;
        unsigned int n_dist_calc = 0;
        vector<int> result;
    };

    struct SearchResults {
        vector<SearchResult> results;
        void push_back(const SearchResult& result) { results.emplace_back(result); }
        void push_back(SearchResult&& result) { results.emplace_back(move(result)); }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "time,n_dist_calc";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                log_ofs << result.time << ","
                        << result.n_dist_calc << endl;

                for (const auto& data_id : result.result) {
                    result_ofs << query_id << ","
                               << data_id << endl;
                }

                query_id++;
            }
        }
    };

    template <typename T>
    T& const_off(const T& ref) {
        return const_cast<T&>(ref);
    }

    struct HNSW {
        const int m, m_max_0, ef_construction;
        const double m_l;
        const bool extend_candidates, keep_pruned_connections;

        int enter_node_id;
        vector<Layer> layers;
        map<int, vector<int>> layer_map;
        Dataset<> dataset;

        mt19937 engine;
        uniform_real_distribution<double> unif_dist;

        HNSW(int m, int m_max_0, int ef_construction,
             bool extend_candidates = true, bool keep_pruned_connections = true) :
                m(m), m_max_0(m_max_0), ef_construction(ef_construction),
                m_l(1 / log(1.0 * m)),
                extend_candidates(extend_candidates),
                keep_pruned_connections(keep_pruned_connections),
                engine(42), unif_dist(0.0, 1.0) {}

        auto get_max_layer() const { return layers.size() - 1; }

        const Node& get_enter_node() const { return layers.back()[enter_node_id]; }

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        auto search_layer(const Data<>& query, int start_node_id, int ef, int l_c) const {
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

                for (const auto neighbor_id : nearest_candidate.neighbors) {
                    if (visited[neighbor_id]) continue;
                    visited[neighbor_id] = true;

                    const auto& neighbor = layers[l_c][neighbor_id];
                    const auto dist_from_neighbor =
                            euclidean_distance(query, neighbor.data);
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
                const auto& candidate = layers[l_c][candidate_id];
                candidate_map.erase(candidate_itr);

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
                for (const auto neighbor_id : neighbors) {
                    if (neighbor_id == new_data.id) continue;
                    auto& neighbor = layers[l_c][neighbor_id];

                    // add a bidirectional edge
                    layers[l_c][new_data.id].neighbors.emplace_back(neighbor_id);
                    neighbor.neighbors.emplace_back(new_data.id);

                    const auto m_max = [l_c, m = m, m_max_0 = m_max_0]() {
                        if (l_c == 0) return m_max_0;
                        else return m;
                    }();

                    if (neighbor.neighbors.size() > m_max) {
                        auto new_neighbor_neighbors = select_neighbors_heuristic(
                                neighbor.data, neighbor.neighbors, m_max + 1, l_c);

                        // erase if first result is itself
                        if (new_neighbor_neighbors.front() == neighbor_id)
                            new_neighbor_neighbors.erase(new_neighbor_neighbors.begin());

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

            auto start_node_id_layer = enter_node_id;
            for (int l_c = get_max_layer(); l_c >= 1; --l_c) {
                const auto nn_id_layer = search_layer(query, start_node_id_layer, 1, l_c)[0];
                start_node_id_layer = nn_id_layer;
            }

            const auto candidates = search_layer(query, start_node_id_layer, ef, 0);
            for (const auto& candidate_id : candidates) {
                const auto& candidate = dataset[candidate_id];
                result.result.emplace_back(candidate_id);
                if (result.result.size() >= k) break;
            }

            return result;
        }
    };
}

#endif //HNSW_HNSW_HPP
