//
// Created by Yusuke Arai on 2020/05/08.
//

#include <map>
#include <gtest/gtest.h>
#include <arailib.hpp>
#include <hnsw.hpp>

using namespace arailib;
using namespace hnsw;

auto calc_recall(const SearchResult& result,
                 const vector<int>& gt) {
    int k = result.result.size();
    int n_acc = 0;

    for (const auto& r : result.result) {
        for (const auto& g : gt) {
            if (r == g) ++n_acc;
        }
    }

    return 1.0 * n_acc / (1.0 * k);
}

auto calc_mean(const vector<double>& v) {
    double mean = 0;
    for (const auto& e : v) mean += e;
    return mean / v.size();
}

TEST(hnsw, get_new_node_level) {
    int m = 15;
    auto index = HNSW(m);

    map<int, int> hash_table;
    for (int i = 0; i < 1000000; i++) {
        const auto l = index.get_new_node_level();
        ++hash_table[l];
    }

    ASSERT_EQ(hash_table[4], 24);
}

TEST(hnsw, knn_search) {
    const string data_path = "/home/arai/workspace/dataset/sift/data1m.csv",
        query_path = "/home/arai/workspace/dataset/sift/sift_query.csv";
    const int n = 100, n_query = 100;

    const auto series = load_data(data_path, n);
    const auto queries = load_data(query_path, n_query);

    int m = 15;
    auto index = HNSW(m);
    index.build(series);

    int k = 10, ef = 20;
    vector<double> recalls(n_query);
#pragma omp parallel for
    for (int i = 0; i < n_query; i++) {
        const auto& query = queries[i];
        const auto result = index.knn_search(query, k, ef);
        const auto scan_result = scan_knn_search(query, k, series);
        const auto recall = calc_recall(result, scan_result);
        recalls[i] = recall;
    }

    cout << calc_mean(recalls) << endl;
}

