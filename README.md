# HNSW
## Overview
It is implementation of approximate kNN search on HNSW: Hierarchical Navigable Small World.

The main algorithm is written `include/hnsw.hpp` (header only), so you can use it easily.

Reference: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (TPAMI 2016)

## Example
```
#include <iostream>
#include <arailib.hpp>
#include <hnsw.hpp>

using namespace std;
using namespace arailib;
using namespace hnsw;

int main() {
    const string data_path = "path/to/data.csv";
    const string query_path = "path/to/query.csv";
    const string save_path = "path/to/result.csv";
    const unsigned n = 1000; // data size
    const unsigned n_query = 1; // query size
    const int k = 10; // result size
    const int ef = 15; // candidate set size
    const int m = 15; // degree

    const auto queries = load_data(query_path, n_query); // read query file

    auto hnsw = HNSW(m); // init HNSW index
    const auto dataset = load_data(data_path, n); // load data
    hnsw.build(dataset); // build index
    
    vector<SearchResult> results(queries.size());
    for (const auto& query : queries) {
        results[query.id] = hnsw.knn_search(query, k, ef);
    }
}
```

## Input File Format
If you want to create index with this three vectors, `(0, 1), (2, 4), (3, 3)`, you must describe data.csv like following format:
```
0,1
2,4
3,3
```

Query format is same as data format.
