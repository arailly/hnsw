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
        vector<reference_wrapper<const Node>> neighbors;
        Node* lower_layer_node;

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

        Node* enter_node;
        vector<Layer> layers;

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

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        RefNodes search_layer(const Data<>& query, const Node& start_node, int ef) const {
            unordered_map<int, bool> visited;
            visited[start_node.data.id] = true;

            const auto dist_from_en = euclidean_distance(query, start_node.data);

            RefNodeMap candidate_map;
            candidate_map.emplace(dist_from_en, start_node);

            RefNodeMap result_map;
            result_map.emplace(dist_from_en, start_node);

            while (!candidate_map.empty()) {
                const auto nearest_candidate = *candidate_map.cbegin();
                candidate_map.erase(candidate_map.cbegin());

                const auto furthest_result = --result_map.cend();

                if (nearest_candidate.first > furthest_result->first) break;

                for (const auto& neighbor : nearest_candidate.second.get().neighbors) {
                    if (visited[neighbor.get().data.id]) continue;
                    visited[neighbor.get().data.id] = true;

                    const auto dist_from_neighbor =
                            euclidean_distance(query, neighbor.get().data);
                    const auto furthest_result_ = --result_map.cend();

                    if (dist_from_neighbor < furthest_result_->first) {
                        candidate_map.emplace(dist_from_neighbor, neighbor);
                        result_map.emplace(dist_from_neighbor, neighbor);

                        if (result_map.size() > ef)
                            result_map.erase(furthest_result_);
                    }
                }
            }

            RefNodes result;
            for (const auto& result_pair : result_map)
                result.emplace_back(result_pair.second);
            return result;
        }

        void insert(const Data<>& new_data) {
            const auto l_new_node = get_new_node_level();

            Node* start_node = enter_node;
            for (int l_c = get_max_layer(); l_c > l_new_node; --l_c) {
                start_node = search_layer(
                        new_data, *start_node, 1)[0].get().lower_layer_node;
            }

            auto new_node_layer = new Node(new_data);
            for (int l_c = min(l_new_node, (int)get_max_layer()); l_c >= 0; --l_c) {
                const auto knn_layer = search_layer(new_data, *start_node, m);
                for (const auto& neighbor : knn_layer) {
                    new_node_layer->neighbors.emplace_back(neighbor);

                    auto& mutable_neighbor = const_off(neighbor.get());
                    mutable_neighbor.neighbors.emplace_back(*new_node_layer);

                    const auto m_max = [l_c, m = m, m_max_0 = m_max_0]() {
                        if (l_c == 0) return m_max_0;
                        else return m;
                    }();

                    if (mutable_neighbor.neighbors.size() > m_max) {
                        const auto new_neighbor_neighbors = search_layer(
                                mutable_neighbor.data, *start_node, m_max);
                        mutable_neighbor.neighbors = new_neighbor_neighbors;
                    }
                }
                layers[l_c].emplace_back(move(*new_node_layer));
                new_node_layer = layers[l_c].back().lower_layer_node;
                start_node = knn_layer[0].get().lower_layer_node;
            }

            if (l_new_node > get_max_layer()) {
                layers.resize(l_new_node + 1);
                layers[l_new_node].emplace_back(new_data);
                enter_node = &layers[l_new_node][0];
            }
        }

        void build(const Series<>& series) {
            for (const auto& data : series) insert(data);
        }

        SearchResult knn_search(const Data<>& query, int k, int ef) const {
            SearchResult result;

            auto start_node_layer = enter_node;
            for (int i = get_max_layer(); i >= 1; --i) {
                start_node_layer = search_layer(
                        query, *start_node_layer, 1)[0].get().lower_layer_node;
            }

            const auto candidates = search_layer(query, *start_node_layer, ef);
            for (const auto& candidate : candidates) {
                const auto dist = euclidean_distance(query, candidate.get().data);
                result.result.emplace(dist, candidate.get().data);
            }

            return result;
        }
    };
}

#endif //HNSW_HNSW_HPP
