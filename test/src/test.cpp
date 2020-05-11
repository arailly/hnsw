//
// Created by Yusuke Arai on 2020/05/08.
//

#include <map>
#include <gtest/gtest.h>
#include <hnsw.hpp>

using namespace arailib;
using namespace hnsw;

TEST(hnsw, get_new_node_level) {
    int m = 15;
    auto index = HNSW(m, m, m);

    map<int, int> hash_table;
    for (int i = 0; i < 1000000; i++) {
        const auto l = index.get_new_node_level();
        ++hash_table[l];
    }

    ASSERT_EQ(hash_table[4], 24);
}

TEST(hnsw, knn_search) {
    const string data_path = "/home/arai/workspace/dataset/sift/data1m/",
        query_path = "/home/arai/workspace/dataset/sift/sift_query.csv";
    const int n = 2, n_query = 100;

    const auto series = load_data(data_path, n);
    const auto queries = load_data(query_path, n_query);

    int m = 5;
    auto index = HNSW(m, m * 2, m);
    index.build(series);

    int k = 5, ef = 10;
    const auto result = index.knn_search(queries[0], k, ef);
    ASSERT_EQ(result.result.size(), k);

    const auto scan_result = scan_knn_search(queries[0], k, series);

    int i = 0;
    for (const auto& e : result.result) {
        ASSERT_EQ(e.second.get(), scan_result[i].get());
        ++i;
    }
}

