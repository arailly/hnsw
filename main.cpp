#include <arailib.hpp>
#include <hnsw.hpp>

using namespace arailib;
using namespace hnsw;

int main() {
    const string base_dir = "/home/arai/workspace/";
    const string data_path = base_dir + "dataset/sift/data1m/",
            query_path = base_dir + "dataset/sift/sift_query.csv";
    const int n = 1000, n_query = 10000;

    const auto series = load_data(data_path, n);
    const auto queries = load_data(query_path, n_query);

    int m = 15;
    auto index = HNSW(m);
    index.build(series);

    int k = 10, ef = 15;
    SearchResults results(n_query);
#pragma omp parallel for
    for (int i = 0; i < n_query; i++) {
        const auto& query = queries[i];
        const auto result = index.knn_search(query, k, ef);
        results[i] = result;
    }

    const string save_name = "m15-ef15.csv";
    const string result_base_dir = base_dir + "result/knn-search/hnsw/sift/data1m/k10/";
    const string log_path = result_base_dir + "log-" + save_name;
    const string result_path = result_base_dir + "result-" + save_name;
    results.save(log_path, result_path);
}
