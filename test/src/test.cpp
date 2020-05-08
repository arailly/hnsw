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