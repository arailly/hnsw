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

    using RefNodes = vector<reference_wrapper<const Node>>;
    using RefNodeMap = map<float, reference_wrapper<const Node>>;
    using Layer = vector<Node>;

    struct SearchResult {
        time_t time = 0;
        unsigned int n_dist_calc = 0;
        map<float, reference_wrapper<const Data<>>> result;
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
            line = "query_id,data_id,distance";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                log_ofs << result.time << ","
                        << result.n_dist_calc << endl;

                for (const auto& result_pair : result.result) {
                    result_ofs << query_id << ","
                               << result_pair.second.get().id << ","
                               << result_pair.first << endl;
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
        const float m_l;
        const bool extend_candidates, keep_pruned_connections;

        int enter_node_id;
        vector<Layer> layers;
        const Series<>* series;

        mt19937 engine;
        uniform_real_distribution<float> unif_dist;

        HNSW(int m, int m_max_0, int ef_construction,
             bool extend_candidates = false, bool keep_pruned_connections = false) :
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

            map<float, int> candidate_map;
            candidate_map.emplace(dist_from_en, start_node_id);

            map<float, int> result_map;
            result_map.emplace(dist_from_en, start_node_id);

            while (!candidate_map.empty()) {
                const auto [dist_from_nearest_candidate, nearest_candidate_id] =
                        *candidate_map.cbegin();
                const auto& nearest_candidate = layers[l_c][nearest_candidate_id];
                candidate_map.erase(candidate_map.cbegin());

                const auto [dist_from_furthest_result, furthest_result_id] =
                        *(--result_map.cend());

                if (dist_from_nearest_candidate > dist_from_furthest_result) break;

                for (const auto& neighbor_id : nearest_candidate.neighbors) {
                    if (visited[neighbor_id]) continue;
                    visited[neighbor_id] = true;

                    const auto& neighbor = layers[l_c][neighbor_id];
                    const auto dist_from_neighbor =
                            euclidean_distance(query, neighbor.data);
                    const auto [dist_from_furthest_result_, furthest_result_id_] =
                            *(--result_map.cend());

                    if (dist_from_neighbor < dist_from_furthest_result_) {
                        candidate_map.emplace(dist_from_neighbor, neighbor_id);
                        result_map.emplace(dist_from_neighbor, neighbor_id);

                        if (result_map.size() > ef)
                            result_map.erase(--result_map.cend());
                    }
                }
            }

            vector<int> result;
            for (const auto& result_pair : result_map)
                result.emplace_back(result_pair.second);
            return result;
        }

        void insert(int new_data_id) {
            const auto& new_data = (*series)[new_data_id];
            const auto l_new_node = get_new_node_level();

            if (layers.empty() || l_new_node > get_max_layer()) {
                // add new layer
                int l_c = get_max_layer() + 1;
                layers.resize(l_new_node + 1);
                for (; l_c <= l_new_node; ++l_c) {
                    for (const auto& data : *series) {
                        layers[l_c].emplace_back(data);
                    }
                }
                enter_node_id = new_data_id;
            } else {
                layers[l_new_node].emplace_back(new_data);
            }

            auto start_node_id = enter_node_id;
            for (int l_c = get_max_layer(); l_c > l_new_node; --l_c) {
                const auto nn_id_layer = search_layer(new_data, start_node_id, 1, l_c)[0];
                start_node_id = nn_id_layer;
            }

            for (int l_c = l_new_node; l_c >= 0; --l_c) {
                const auto knn_layer = search_layer(new_data, start_node_id, m, l_c);
                for (const auto& neighbor_id : knn_layer) {
                    if (neighbor_id == new_data_id) continue;
                    layers[l_c][new_data_id].neighbors.emplace_back(neighbor_id);

                    auto& neighbor = layers[l_c][neighbor_id];
                    neighbor.neighbors.emplace_back(new_data_id);

                    const auto m_max = [l_c, m = m, m_max_0 = m_max_0]() {
                        if (l_c == 0) return m_max_0;
                        else return m;
                    }();

                    if (neighbor.neighbors.size() > m_max) {
                        auto new_neighbor_knn = search_layer(
                                neighbor.data, start_node_id, m_max + 1, l_c);
                        // first result is itself
                        new_neighbor_knn.erase(new_neighbor_knn.begin());

                        if (new_neighbor_knn.size() >= m_max)
                            neighbor.neighbors = new_neighbor_knn;
                        else
                            neighbor.neighbors.erase(--neighbor.neighbors.cend());
                    }
                }
                if (l_c == 0) continue;
                start_node_id = knn_layer[0];
            }
        }

        void build(const Series<>& series_) {
            series = &series_;
            for (const auto& data : series_) insert(data.id);
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
                const auto& candidate = (*series)[candidate_id];
                const auto dist = euclidean_distance(query, candidate);
                result.result.emplace(dist, candidate);
                if (result.result.size() >= k) break;
            }

            return result;
        }
    };
}

#endif //HNSW_HNSW_HPP
