//
// Created by Yusuke Arai on 2020/05/08.
//

#ifndef HNSW_HNSW_HPP
#define HNSW_HNSW_HPP

#include <queue>
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

    struct Candidate {
        double dist;
        int id;

        Candidate() : dist(-1), id(-1) {}
        Candidate(double dist, int id) : dist(dist), id(id) {}
    };

    struct CompLess {
        constexpr bool operator()(const Candidate& c1, const Candidate& c2) const noexcept {
            return c1.dist < c2.dist;
        }
    };

    struct CompGreater {
        constexpr bool operator()(const Candidate& c1, const Candidate& c2) const noexcept {
            return c1.dist > c2.dist;
        }
    };

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

        HNSW(int m, int ef_construction = 50,
             bool extend_candidates = true, bool keep_pruned_connections = true) :
                m(m), m_max_0(m * 2), m_l(1 / log(1.0 * m)),
                ef_construction(ef_construction),
                extend_candidates(extend_candidates),
                keep_pruned_connections(keep_pruned_connections),
                engine(42), unif_dist(0.0, 1.0) {}

        auto get_max_layer() const { return layers.size() - 1; }

        const Node& get_enter_node() const { return layers.back()[enter_node_id]; }

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        auto search_layer(const Data<>& query, int start_node_id, int ef, int l_c) {
            auto result = SearchResult();

            vector<bool> visited(dataset.size());
            visited[start_node_id] = true;

            priority_queue<Candidate, vector<Candidate>, CompGreater> candidates;
            priority_queue<Candidate, vector<Candidate>, CompLess> top_candidates;

            const auto& start_node = layers[l_c][start_node_id];
            const auto dist_from_en = euclidean_distance(query, start_node.data);

            candidates.emplace(dist_from_en, start_node_id);
            top_candidates.emplace(dist_from_en, start_node_id);

            while (!candidates.empty()) {
                const auto nearest_candidate = candidates.top();
                const auto& nearest_candidate_node = layers[l_c][nearest_candidate.id];
                candidates.pop();

                if (nearest_candidate.dist > top_candidates.top().dist) break;

                ++result.n_hop;

                for (const auto neighbor_id : nearest_candidate_node.neighbors) {
                    if (visited[neighbor_id]) continue;
                    visited[neighbor_id] = true;

                    const auto& neighbor = layers[l_c][neighbor_id];
                    const auto dist_from_neighbor =
                            euclidean_distance(query, neighbor.data);
                    ++result.n_dist_calc;

                    if (dist_from_neighbor < top_candidates.top().dist ||
                        top_candidates.size() < ef) {
                        candidates.emplace(dist_from_neighbor, neighbor_id);
                        top_candidates.emplace(dist_from_neighbor, neighbor_id);

                        if (top_candidates.size() > ef) top_candidates.pop();
                    }
                }
            }

            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top().id);
                top_candidates.pop();
            }

            reverse(result.result.begin(), result.result.end());

            return result;
        }

        auto select_neighbors_heuristic(const Data<>& query, vector<int> initial_candidates,
                                        int n_neighbors, int l_c) {
            priority_queue<Candidate, vector<Candidate>, CompLess> top_candidates;
            priority_queue<Candidate, vector<Candidate>, CompGreater>
                    candidates, discarded_candidates;

            vector<bool> added(dataset.size());
            added[query.id] = true;

            for (const auto& candidate_id : initial_candidates) {
                if (added[candidate_id]) continue;
                added[candidate_id] = true;

                const auto& candidate_node = layers[l_c][candidate_id];
                const auto dist_from_candidate =
                        euclidean_distance(query, candidate_node.data);
                candidates.emplace(dist_from_candidate, candidate_id);
            }

            if (extend_candidates) {
                for (const auto& candidate_id : initial_candidates) {
                    if (added[candidate_id]) continue;
                    added[candidate_id] = true;

                    // candidate must be get like below (not to get ref of layers[l_c])
                    const auto& candidate_node = layers[l_c][candidate_id];
                    for (const auto& neighbor_id : candidate_node.neighbors) {
                        const auto& neighbor = layers[l_c][neighbor_id];
                        const auto dist_from_neighbor =
                                euclidean_distance(query, neighbor.data);
                        candidates.emplace(dist_from_neighbor, neighbor_id);
                    }
                }
            }

            while (!candidates.empty() && top_candidates.size() < n_neighbors) {
                const auto candidate = candidates.top();
                candidates.pop();

                const auto& candidate_node = layers[l_c][candidate.id];

                if (top_candidates.empty()) {
                    top_candidates.emplace(candidate);
                    continue;
                }

                if (candidate.dist < top_candidates.top().dist)
                    top_candidates.emplace(candidate);
                else
                    discarded_candidates.emplace(candidate);
            }

            if (keep_pruned_connections) {
                while (!discarded_candidates.empty()
                       && top_candidates.size() < n_neighbors) {
                    top_candidates.emplace(discarded_candidates.top());
                    discarded_candidates.pop();
                }
            }

            vector<int> result;
            while (!top_candidates.empty()) {
                result.emplace_back(top_candidates.top().id);
                top_candidates.pop();
            }
            reverse(result.begin(), result.end());

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
                const auto nn_id_layer = search_layer(
                        new_data, start_node_id, 1, l_c).result[0];
                start_node_id = nn_id_layer;
            }

            for (int l_c = l_new_node; l_c >= 0; --l_c) {
                auto neighbors = search_layer(
                        new_data, start_node_id, ef_construction, l_c).result;
                if (neighbors.size() > m) neighbors.resize(m);

                auto& layer = layers[l_c];
                for (const auto neighbor_id : neighbors) {
                    if (neighbor_id == new_data.id) continue;
                    auto& neighbor = layer[neighbor_id];

                    // add a bidirectional edge
                    layer[new_data.id].neighbors.emplace_back(neighbor_id);
                    neighbor.neighbors.emplace_back(new_data.id);

//                    const auto m_max = l_c == 0 ? m_max_0 : m;
//
//                    if (neighbor.neighbors.size() > m_max) {
//                        const auto new_neighbor_neighbors = select_neighbors_heuristic(
//                                neighbor.data, neighbor.neighbors, m_max, l_c);
//                        neighbor.neighbors = new_neighbor_neighbors;
//                    }
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
            cout << "complete: build" << endl;
        }

        auto knn_search(const Data<>& query, int k, int ef) {
            SearchResult result;
            const auto begin = get_now();

            // search in upper layers
            auto start_id_layer = enter_node_id;
            for (int l_c = get_max_layer(); l_c >= 1; --l_c) {
                const auto result_layer = search_layer(
                        query, start_id_layer, 1, l_c);

                result.n_hop_upper_layer += result_layer.n_hop;
                result.n_dist_calc_upper_layer += result_layer.n_dist_calc;

                const auto& nn_id_layer = result_layer.result[0];
                start_id_layer = nn_id_layer;
            }

            const auto& nn_upper_layer = layers[1][start_id_layer];

            // search in base layer
            const auto result_layer = search_layer(query, start_id_layer, ef, 0);
            const auto candidates = result_layer.result;
            for (const auto& candidate_id : candidates) {
                const auto& candidate = dataset[candidate_id];
                result.result.emplace_back(candidate_id);
                if (result.result.size() >= k) break;
            }

            const auto end = get_now();
            result.time = get_duration(begin, end);

            result.n_dist_calc_base_layer = result_layer.n_dist_calc;
            result.n_dist_calc = result.n_dist_calc_upper_layer + result.n_dist_calc_base_layer;

            result.n_hop_base_layer = result_layer.n_hop;
            result.n_hop = result.n_hop_upper_layer + result.n_hop_base_layer;

            result.dist_from_ep_base_layer = euclidean_distance(query, nn_upper_layer.data);

            return result;
        }
    };
}

#endif //HNSW_HNSW_HPP
