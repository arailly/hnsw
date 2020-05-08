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
        Node* next_layer_node;

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

    struct HNSW {
        const int m, m_max, ef_construction;
        const float m_l;
        const bool extend_candidates, keep_pruned_connections;

        Node* enter_node;
        vector<Layer> layers;
        Series<> series;

        mt19937 engine;
        uniform_real_distribution<float> unif_dist;

        HNSW(int m, int m_max, int ef_construction,
             bool extend_candidates = false, bool keep_pruned_connections = false) :
                m(m), m_max(m_max), ef_construction(ef_construction),
                m_l(1 / log(1.0 * m)),
                extend_candidates(extend_candidates),
                keep_pruned_connections(keep_pruned_connections),
                engine(42), unif_dist(0.0, 1.0) {}

        int get_new_node_level() {
            return static_cast<int>(-log(unif_dist(engine)) * m_l);
        }

        RefNodes search_layer(const Node& query, const Node& start_node, int ef) {
            unordered_map<int, bool> visited;
            visited[start_node.data.id] = true;

            const auto dist_from_en = euclidean_distance(query.data, start_node.data);

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
                            euclidean_distance(query.data, neighbor.get().data);
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
            const auto new_node = Node(new_data);
            const auto new_node_level = get_new_node_level();

            if (new_node_level > layers.size() - 1)
                layers.resize(new_node_level + 1);
        }
    };
}

#endif //HNSW_HNSW_HPP
