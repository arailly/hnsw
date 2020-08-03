#include <arailib.hpp>
#include <hnsw.hpp>

using namespace arailib;
using namespace hnsw;

int main() {
    const string base_dir = "/home/arai/workspace/";
    const string data_path = base_dir + "dataset/sift/data1m/",
            query_path = base_dir + "dataset/sift/query10k.csv",
            ground_truth_path = base_dir + "result/knn-search/scan/sift/data1m/k10/result.csv";
    const int n = 1000, n_query = 10000;

    const auto dataset = load_data(data_path, n);
    const auto queries = load_data(query_path, n_query);
    const auto ground_truth = load_neighbors(ground_truth_path, n_query, true);

    const auto start = get_now();

    int m = 10;
    auto index = HNSW(m);
    index.build(dataset);

    const auto end = get_now();
    const auto build_time = get_duration(start, end);

    cout << "complete: build index" << endl;
    cout << "indexing time: " << build_time / 60000000 << " [min]" << endl;

    int k = 10;

    for (int ef = k; ef <=40; ef += 5) {
        SearchResults results(n_query);
        for (int i = 0; i < n_query; i++) {
            const auto& query = queries[i];
            auto result = index.knn_search(query, k, ef);
            result.recall = calc_recall(result.result, ground_truth[query.id]);
            results[i] = result;
        }

        const string save_name = "m10-ef" + to_string(ef) + ".csv";
        const string result_base_dir = base_dir + "result/knn-search/hnsw/sift/data1m/k10/";
        const string log_path = result_base_dir + "log-" + save_name;
        const string result_path = result_base_dir + "result-" + save_name;
        results.save(log_path, result_path);
    }
}
