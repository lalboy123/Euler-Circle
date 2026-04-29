#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <tuple>
#include <omp.h>
#include <mutex>

std::mutex mtx; 

using namespace std;

// --- Global Constants ---
const int MAX_N = 15;
const int MAX_K = 10; // Increased to 10 and used in Summary
const int MAX_MASK = 128;

// --- Core Algorithm Translations ---
struct Summary {
    int min_v[MAX_K] = {0};
    int max_v[MAX_K] = {0};

    bool operator<(const Summary& o) const {
        for (int i = 0; i < MAX_K; ++i) {
            if (min_v[i] != o.min_v[i]) return min_v[i] < o.min_v[i];
            if (max_v[i] != o.max_v[i]) return max_v[i] < o.max_v[i];
        }
        return false;
    }
    bool operator==(const Summary& o) const {
        for (int i = 0; i < MAX_K; ++i) {
            if (min_v[i] != o.min_v[i] || max_v[i] != o.max_v[i]) return false;
        }
        return true;
    }
};

// --- Thread-Local Storage (Consolidated here to avoid multiple definitions) ---
thread_local bool dp_visited[MAX_N][MAX_K][MAX_MASK];
thread_local vector<tuple<int, int, int>> dp_touched;
thread_local vector<Summary> dp_memo[MAX_N][MAX_K][MAX_MASK];

// --- Math Helpers ---
long long stirling2(int n, int t) {
    if (t < 0 || t > n) return 0;
    if (n == 0 && t == 0) return 1;
    if (t == 0) return 0;
    vector<vector<long long>> S(n + 1, vector<long long>(t + 1, 0));
    S[0][0] = 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= min(i, t); j++) {
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1];
        }
    }
    return S[n][t];
}

long long nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    long long res = 1;
    for (int i = 1; i <= r; i++) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

long long fact(int n) {
    long long res = 1;
    for (int i = 1; i <= n; i++) res *= i;
    return res;
}

long long power(long long base, int exp) {
    long long res = 1;
    while (exp > 0) {
        if (exp % 2 == 1) res *= base;
        base *= base;
        exp /= 2;
    }
    return res;
}

long long expression(int n, int t) {
    long long total = 0;
    for (int j = n; j < t; j++) {
        long long term1 = power(n, t - 1 - j);
        long long term2 = (j - n + 1);
        long long term3 = nCr(j, n - 1);
        long long term4 = fact(j - n + 1);
        long long s_val = stirling2(j - 2, j - n + 1);
        total += term1 * term2 * term3 * term4 * s_val;
    }
    for (int L = 0; L < n; L++) {
        long long term1 = nCr(t, L);
        long long term2 = fact(t - L);
        long long s_val = stirling2(t - 2, t - L);
        total += term1 * term2 * s_val;
    }
    return total;
}

bool is_star(const vector<int>& seq) {
    if (seq.size() < 2) return false;
    int k = seq.size() + 2;
    bool all_k = true, all_1 = true;
    for (int x : seq) {
        if (x != k) all_k = false;
        if (x != 1) all_1 = false;
    }
    return all_k || all_1;
}

vector<int> get_symmetric(const vector<int>& seq) {
    int n = seq.size();
    vector<int> sym(n);
    for (int i = 0; i < n; ++i) {
        sym[i] = n + 3 - seq[n - 1 - i];
    }
    return sym;
}

vector<vector<int>> all_prufer_sequences(int n) {
    vector<vector<int>> res;
    if (n == 2) { res.push_back({}); return res; }
    vector<int> seq(n - 2, 1);
    while (true) {
        res.push_back(seq);
        int ptr = n - 3;
        while (ptr >= 0 && seq[ptr] == n) {
            seq[ptr] = 1;
            ptr--;
        }
        if (ptr < 0) break;
        seq[ptr]++;
    }
    return res;
}

vector<pair<int, int>> prufer_to_tree(const vector<int>& prufer, int size) {
    if (size == 1) return {};
    if (size == 2) return {{1, 2}};
    vector<int> degree(size + 1, 1);
    for (int node : prufer) degree[node]++;
    priority_queue<int, vector<int>, greater<int>> leaves;
    for (int i = 1; i <= size; ++i) {
        if (degree[i] == 1) leaves.push(i);
    }
    vector<pair<int, int>> edges;
    for (int node : prufer) {
        int leaf = leaves.top(); leaves.pop();
        edges.push_back({leaf, node});
        degree[node]--;
        if (degree[node] == 1) leaves.push(node);
    }
    int u = leaves.top(); leaves.pop();
    int v = leaves.top(); leaves.pop();
    edges.push_back({u, v});
    return edges;
}

bool satisfies_condition(const Summary& s, int k) {
    int pref_max[MAX_K] = {0};
    int curr_max = -1;
    for (int i = 0; i < k; ++i) {
        if (s.min_v[i] != 0 && s.min_v[i] > curr_max) curr_max = s.min_v[i];
        pref_max[i] = curr_max;
    }
    for (int j = 1; j < k; ++j) {
        if (s.min_v[j] != 0 && pref_max[j - 1] >= s.max_v[j]) return false;
    }
    return true;
}

// --- Recursive Solver ---
vector<Summary> solve(int u, int p, int R_mask, const vector<vector<int>>& children_T, const vector<vector<int>>& children_P, int k) {
    if (dp_visited[u][p][R_mask]) return dp_memo[u][p][R_mask];
    
    dp_visited[u][p][R_mask] = true;
    dp_touched.push_back({u, p, R_mask});
    vector<Summary>& res = dp_memo[u][p][R_mask];
    res.clear();

    int c_len = children_T[u].size();
    if (c_len == 0) {
        if (R_mask == 0) {
            Summary base;
            base.min_v[p - 1] = u;
            base.max_v[p - 1] = u;
            res.push_back(base);
        }
        return res;
    }

    vector<vector<vector<Summary>>> child_options(c_len, vector<vector<Summary>>(MAX_MASK));
    vector<vector<int>> valid_masks_for_child(c_len);

    for (int i = 0; i < c_len; ++i) {
        int v = children_T[u][i];
        int S = R_mask;
        do {
            vector<Summary> opts;
            if (__builtin_popcount(S) == 1) {
                int c = __builtin_ctz(S) + 1;
                int c_children_mask = 0;
                for (int child : children_P[c]) c_children_mask |= (1 << (child - 1));
                vector<Summary> sub_opts = solve(v, c, c_children_mask, children_T, children_P, k);
                opts.insert(opts.end(), sub_opts.begin(), sub_opts.end());
            }
            vector<Summary> sub_opts2 = solve(v, p, S, children_T, children_P, k);
            opts.insert(opts.end(), sub_opts2.begin(), sub_opts2.end());

            if (!opts.empty()) {
                sort(opts.begin(), opts.end());
                opts.erase(unique(opts.begin(), opts.end()), opts.end());
                child_options[i][S] = opts;
                valid_masks_for_child[i].push_back(S);
            }
            if (S == 0) break;
            S = (S - 1) & R_mask;
        } while (S != R_mask);
    }

    vector<int> available_masks(c_len, 0);
    int curr_or = 0;
    for (int i = c_len - 1; i >= 0; --i) {
        for (int mask : valid_masks_for_child[i]) curr_or |= mask;
        available_masks[i] = curr_or;
    }

    Summary base;
    base.min_v[p - 1] = u;
    base.max_v[p - 1] = u;
    vector<Summary> all_mappings;
    
    auto combine = [&](auto& self, int child_idx, int current_mask, vector<Summary>& current_mappings) -> void {
        if (child_idx == c_len) {
            if (current_mask == R_mask) {
                Summary merged = base;
                for (auto& m : current_mappings) {
                    for (int i = 0; i < k; ++i) {
                        if (m.min_v[i] != 0) {
                            if (merged.min_v[i] == 0) {
                                merged.min_v[i] = m.min_v[i];
                                merged.max_v[i] = m.max_v[i];
                            } else {
                                if (m.min_v[i] < merged.min_v[i]) merged.min_v[i] = m.min_v[i];
                                if (m.max_v[i] > merged.max_v[i]) merged.max_v[i] = m.max_v[i];
                            }
                        }
                    }
                }
                all_mappings.push_back(merged);
            }
            return;
        }
        if ((current_mask | available_masks[child_idx]) != R_mask) return;
        for (int mask : valid_masks_for_child[child_idx]) {
            if ((current_mask & mask) == 0) {
                for (auto& opt : child_options[child_idx][mask]) {
                    current_mappings.push_back(opt);
                    self(self, child_idx + 1, current_mask | mask, current_mappings);
                    current_mappings.pop_back();
                }
            }
        }
    };

    vector<Summary> curr_maps;
    combine(combine, 0, 0, curr_maps);
    sort(all_mappings.begin(), all_mappings.end());
    all_mappings.erase(unique(all_mappings.begin(), all_mappings.end()), all_mappings.end());
    res = all_mappings;
    return res;
}

// --- Main Interface Function ---
bool is_minor(const vector<int>& prufer_T, const vector<vector<int>>& children_P, int root_P_mask, int k) {
    int n = (int)prufer_T.size() + 2;
    if (k > n) return false;

    static thread_local vector<vector<int>> adj_T;
    static thread_local vector<vector<int>> children_T;
    static thread_local vector<bool> visited_T;
    static thread_local queue<int> qt;

    adj_T.assign(n + 1, vector<int>());
    auto edges_T = prufer_to_tree(prufer_T, n);
    for (auto& e : edges_T) {
        adj_T[e.first].push_back(e.second);
        adj_T[e.second].push_back(e.first);
    }

    for (int r_T = 1; r_T <= n; ++r_T) {
        children_T.assign(n + 1, vector<int>());
        visited_T.assign(n + 1, false);
        while(!qt.empty()) qt.pop();
        
        qt.push(r_T);
        visited_T[r_T] = true;
        while (!qt.empty()) {
            int curr = qt.front(); qt.pop();
            for (int nxt : adj_T[curr]) {
                if (!visited_T[nxt]) {
                    visited_T[nxt] = true;
                    children_T[curr].push_back(nxt);
                    qt.push(nxt);
                }
            }
        }

        for (auto& t : dp_touched) {
            dp_visited[get<0>(t)][get<1>(t)][get<2>(t)] = false;
        }
        dp_touched.clear();

        vector<Summary> mappings = solve(r_T, 1, root_P_mask, children_T, children_P, k);
        for (const auto& summary : mappings) {
            if (satisfies_condition(summary, k)) return true;
        }
    }
    return false;
}

// --- Hardcoded Data Extraction ---
map<pair<vector<int>, int>, long long> static_table;
void load_table() {
    static_table[{ {}, 2 }] = 0LL;
    static_table[{ {}, 3 }] = 0LL;
    static_table[{ {}, 4 }] = 0LL;
    static_table[{ {}, 5 }] = 0LL;
    static_table[{ {}, 6 }] = 0LL;
    static_table[{ {}, 7 }] = 0LL;
    static_table[{ {}, 8 }] = 0LL;
    static_table[{ {1}, 3 }] = 2LL;
    static_table[{ {1}, 4 }] = 4LL;
    static_table[{ {1}, 5 }] = 8LL;
    static_table[{ {1}, 6 }] = 16LL;
    static_table[{ {1}, 7 }] = 32LL;
    static_table[{ {1}, 8 }] = 64LL;
    static_table[{ {2}, 3 }] = 2LL;
    static_table[{ {2}, 4 }] = 0LL;
    static_table[{ {2}, 5 }] = 0LL;
    static_table[{ {2}, 6 }] = 0LL;
    static_table[{ {2}, 7 }] = 0LL;
    static_table[{ {2}, 8 }] = 0LL;
    static_table[{ {3}, 3 }] = 2LL;
    static_table[{ {3}, 4 }] = 4LL;
    static_table[{ {3}, 5 }] = 8LL;
    static_table[{ {3}, 6 }] = 16LL;
    static_table[{ {3}, 7 }] = 32LL;
    static_table[{ {3}, 8 }] = 64LL;
    static_table[{ {1, 1}, 4 }] = 15LL;
    static_table[{ {1, 1}, 5 }] = 93LL;
    static_table[{ {1, 1}, 6 }] = 639LL;
    static_table[{ {1, 1}, 7 }] = 4797LL;
    static_table[{ {1, 1}, 8 }] = 39591LL;
    static_table[{ {1, 2}, 5 }] = 86LL;
    static_table[{ {1, 2}, 6 }] = 501LL;
    static_table[{ {1, 2}, 7 }] = 2945LL;
    static_table[{ {1, 2}, 8 }] = 17452LL;
    static_table[{ {1, 3}, 4 }] = 15LL;
    static_table[{ {1, 3}, 5 }] = 84LL;
    static_table[{ {1, 3}, 6 }] = 409LL;
    static_table[{ {1, 3}, 7 }] = 1926LL;
    static_table[{ {1, 3}, 8 }] = 9325LL;
    static_table[{ {1, 4}, 4 }] = 15LL;
    static_table[{ {1, 4}, 5 }] = 86LL;
    static_table[{ {1, 4}, 6 }] = 437LL;
    static_table[{ {1, 4}, 7 }] = 1937LL;
    static_table[{ {1, 4}, 8 }] = 7675LL;
    static_table[{ {2, 1}, 4 }] = 15LL;
    static_table[{ {2, 1}, 5 }] = 87LL;
    static_table[{ {2, 1}, 6 }] = 528LL;
    static_table[{ {2, 1}, 7 }] = 3244LL;
    static_table[{ {2, 1}, 8 }] = 20000LL;
    static_table[{ {2, 2}, 4 }] = 15LL;
    static_table[{ {2, 2}, 5 }] = 77LL;
    static_table[{ {2, 2}, 6 }] = 431LL;
    static_table[{ {2, 2}, 7 }] = 2839LL;
    static_table[{ {2, 2}, 8 }] = 21803LL;
    static_table[{ {2, 3}, 4 }] = 15LL;
    static_table[{ {2, 3}, 5 }] = 82LL;
    static_table[{ {2, 3}, 6 }] = 254LL;
    static_table[{ {2, 3}, 7 }] = 678LL;
    static_table[{ {2, 3}, 8 }] = 1664LL;
    static_table[{ {2, 4}, 4 }] = 15LL;
    static_table[{ {2, 4}, 5 }] = 84LL;
    static_table[{ {2, 4}, 6 }] = 409LL;
    static_table[{ {2, 4}, 7 }] = 1926LL;
    static_table[{ {2, 4}, 8 }] = 9325LL;
    static_table[{ {3, 1}, 4 }] = 15LL;
    static_table[{ {3, 1}, 5 }] = 85LL;
    static_table[{ {3, 1}, 6 }] = 460LL;
    static_table[{ {3, 1}, 7 }] = 2349LL;
    static_table[{ {3, 1}, 8 }] = 11547LL;
    static_table[{ {3, 2}, 4 }] = 15LL;
    static_table[{ {3, 2}, 5 }] = 80LL;
    static_table[{ {3, 2}, 6 }] = 305LL;
    static_table[{ {3, 2}, 7 }] = 1072LL;
    static_table[{ {3, 2}, 8 }] = 3522LL;
    static_table[{ {3, 3}, 4 }] = 15LL;
    static_table[{ {3, 3}, 5 }] = 77LL;
    static_table[{ {3, 3}, 6 }] = 431LL;
    static_table[{ {3, 3}, 7 }] = 2839LL;
    static_table[{ {3, 3}, 8 }] = 21803LL;
    static_table[{ {3, 4}, 4 }] = 15LL;
    static_table[{ {3, 4}, 5 }] = 87LL;
    static_table[{ {3, 4}, 6 }] = 528LL;
    static_table[{ {3, 4}, 7 }] = 3244LL;
    static_table[{ {3, 4}, 8 }] = 20000LL;
    static_table[{ {4, 1}, 4 }] = 15LL;
    static_table[{ {4, 1}, 5 }] = 86LL;
    static_table[{ {4, 1}, 6 }] = 437LL;
    static_table[{ {4, 1}, 7 }] = 1937LL;
    static_table[{ {4, 1}, 8 }] = 7675LL;
    static_table[{ {4, 2}, 4 }] = 15LL;
    static_table[{ {4, 2}, 5 }] = 84LL;
    static_table[{ {4, 2}, 6 }] = 409LL;
    static_table[{ {4, 2}, 7 }] = 1926LL;
    static_table[{ {4, 2}, 8 }] = 9325LL;
    static_table[{ {4, 3}, 5 }] = 86LL;
    static_table[{ {4, 3}, 6 }] = 501LL;
    static_table[{ {4, 3}, 7 }] = 2945LL;
    static_table[{ {4, 3}, 8 }] = 17452LL;
    static_table[{ {4, 4}, 4 }] = 15LL;
    static_table[{ {4, 4}, 5 }] = 93LL;
    static_table[{ {4, 4}, 6 }] = 639LL;
    static_table[{ {4, 4}, 7 }] = 4797LL;
    static_table[{ {4, 4}, 8 }] = 39591LL;
    static_table[{ {1, 1, 1}, 5 }] = 124LL;
    static_table[{ {1, 1, 1}, 6 }] = 1216LL;
    static_table[{ {1, 1, 1}, 7 }] = 13624LL;
    static_table[{ {1, 1, 1}, 8 }] = 165376LL;
    static_table[{ {1, 1, 2}, 5 }] = 124LL;
    static_table[{ {1, 1, 2}, 6 }] = 1211LL;
    static_table[{ {1, 1, 2}, 7 }] = 13311LL;
    static_table[{ {1, 1, 2}, 8 }] = 153137LL;
    static_table[{ {1, 1, 3}, 5 }] = 124LL;
    static_table[{ {1, 1, 3}, 6 }] = 1205LL;
    static_table[{ {1, 1, 3}, 7 }] = 12857LL;
    static_table[{ {1, 1, 3}, 8 }] = 140592LL;
    static_table[{ {1, 1, 4}, 5 }] = 124LL;
    static_table[{ {1, 1, 4}, 6 }] = 1207LL;
    static_table[{ {1, 1, 4}, 7 }] = 12903LL;
    static_table[{ {1, 1, 4}, 8 }] = 142345LL;
    static_table[{ {1, 1, 5}, 5 }] = 124LL;
    static_table[{ {1, 1, 5}, 6 }] = 1212LL;
    static_table[{ {1, 1, 5}, 7 }] = 13213LL;
    static_table[{ {1, 1, 5}, 8 }] = 150493LL;
    static_table[{ {1, 2, 1}, 5 }] = 124LL;
    static_table[{ {1, 2, 1}, 6 }] = 1211LL;
    static_table[{ {1, 2, 1}, 7 }] = 13310LL;
    static_table[{ {1, 2, 1}, 8 }] = 153074LL;
    static_table[{ {1, 2, 2}, 5 }] = 124LL;
    static_table[{ {1, 2, 2}, 6 }] = 1201LL;
    static_table[{ {1, 2, 2}, 7 }] = 12392LL;
    static_table[{ {1, 2, 2}, 8 }] = 126600LL;
    static_table[{ {1, 2, 3}, 5 }] = 124LL;
    static_table[{ {1, 2, 3}, 6 }] = 1193LL;
    static_table[{ {1, 2, 3}, 7 }] = 11783LL;
    static_table[{ {1, 2, 3}, 8 }] = 107147LL;
    static_table[{ {1, 2, 4}, 5 }] = 124LL;
    static_table[{ {1, 2, 4}, 6 }] = 1198LL;
    static_table[{ {1, 2, 4}, 7 }] = 12053LL;
    static_table[{ {1, 2, 4}, 8 }] = 114674LL;
    static_table[{ {1, 2, 5}, 5 }] = 124LL;
    static_table[{ {1, 2, 5}, 6 }] = 1209LL;
    static_table[{ {1, 2, 5}, 7 }] = 12954LL;
    static_table[{ {1, 2, 5}, 8 }] = 140555LL;
    static_table[{ {1, 3, 1}, 5 }] = 124LL;
    static_table[{ {1, 3, 1}, 6 }] = 1205LL;
    static_table[{ {1, 3, 1}, 7 }] = 12857LL;
    static_table[{ {1, 3, 1}, 8 }] = 140592LL;
    static_table[{ {1, 3, 2}, 5 }] = 124LL;
    static_table[{ {1, 3, 2}, 6 }] = 1193LL;
    static_table[{ {1, 3, 2}, 7 }] = 11786LL;
    static_table[{ {1, 3, 2}, 8 }] = 107275LL;
    static_table[{ {1, 3, 3}, 5 }] = 124LL;
    static_table[{ {1, 3, 3}, 6 }] = 1184LL;
    static_table[{ {1, 3, 3}, 7 }] = 11181LL;
    static_table[{ {1, 3, 3}, 8 }] = 92601LL;
    static_table[{ {1, 3, 4}, 5 }] = 124LL;
    static_table[{ {1, 3, 4}, 6 }] = 1189LL;
    static_table[{ {1, 3, 4}, 7 }] = 11311LL;
    static_table[{ {1, 3, 4}, 8 }] = 97136LL;
    static_table[{ {1, 3, 5}, 5 }] = 124LL;
    static_table[{ {1, 3, 5}, 6 }] = 1204LL;
    static_table[{ {1, 3, 5}, 7 }] = 12523LL;
    static_table[{ {1, 3, 5}, 8 }] = 125347LL;
    static_table[{ {1, 4, 1}, 5 }] = 124LL;
    static_table[{ {1, 4, 1}, 6 }] = 1207LL;
    static_table[{ {1, 4, 1}, 7 }] = 12903LL;
    static_table[{ {1, 4, 1}, 8 }] = 142345LL;
    static_table[{ {1, 4, 2}, 5 }] = 124LL;
    static_table[{ {1, 4, 2}, 6 }] = 1198LL;
    static_table[{ {1, 4, 2}, 7 }] = 12053LL;
    static_table[{ {1, 4, 2}, 8 }] = 114674LL;
    static_table[{ {1, 4, 3}, 5 }] = 124LL;
    static_table[{ {1, 4, 3}, 6 }] = 1189LL;
    static_table[{ {1, 4, 3}, 7 }] = 11311LL;
    static_table[{ {1, 4, 3}, 8 }] = 97136LL;
    static_table[{ {1, 4, 4}, 5 }] = 124LL;
    static_table[{ {1, 4, 4}, 6 }] = 1192LL;
    static_table[{ {1, 4, 4}, 7 }] = 11440LL;
    static_table[{ {1, 4, 4}, 8 }] = 101869LL;
    static_table[{ {1, 4, 5}, 5 }] = 124LL;
    static_table[{ {1, 4, 5}, 6 }] = 1211LL;
    static_table[{ {1, 4, 5}, 7 }] = 13311LL;
    static_table[{ {1, 4, 5}, 8 }] = 153137LL;
    static_table[{ {1, 5, 1}, 5 }] = 124LL;
    static_table[{ {1, 5, 1}, 6 }] = 1212LL;
    static_table[{ {1, 5, 1}, 7 }] = 13213LL;
    static_table[{ {1, 5, 1}, 8 }] = 150493LL;
    static_table[{ {1, 5, 2}, 5 }] = 124LL;
    static_table[{ {1, 5, 2}, 6 }] = 1209LL;
    static_table[{ {1, 5, 2}, 7 }] = 12954LL;
    static_table[{ {1, 5, 2}, 8 }] = 140555LL;
    static_table[{ {1, 5, 3}, 5 }] = 124LL;
    static_table[{ {1, 5, 3}, 6 }] = 1204LL;
    static_table[{ {1, 5, 3}, 7 }] = 12523LL;
    static_table[{ {1, 5, 3}, 8 }] = 125347LL;
    static_table[{ {1, 5, 4}, 5 }] = 124LL;
    static_table[{ {1, 5, 4}, 6 }] = 1211LL;
    static_table[{ {1, 5, 4}, 7 }] = 13311LL;
    static_table[{ {1, 5, 4}, 8 }] = 153137LL;
    static_table[{ {1, 5, 5}, 5 }] = 124LL;
    static_table[{ {1, 5, 5}, 6 }] = 1216LL;
    static_table[{ {1, 5, 5}, 7 }] = 13624LL;
    static_table[{ {1, 5, 5}, 8 }] = 165376LL;
    static_table[{ {2, 1, 1}, 5 }] = 124LL;
    static_table[{ {2, 1, 1}, 6 }] = 1211LL;
    static_table[{ {2, 1, 1}, 7 }] = 13310LL;
    static_table[{ {2, 1, 1}, 8 }] = 153074LL;
    static_table[{ {2, 1, 2}, 5 }] = 124LL;
    static_table[{ {2, 1, 2}, 6 }] = 1206LL;
    static_table[{ {2, 1, 2}, 7 }] = 12759LL;
    static_table[{ {2, 1, 2}, 8 }] = 139178LL;
    static_table[{ {2, 1, 3}, 5 }] = 124LL;
    static_table[{ {2, 1, 3}, 6 }] = 1198LL;
    static_table[{ {2, 1, 3}, 7 }] = 12053LL;
    static_table[{ {2, 1, 3}, 8 }] = 114674LL;
    static_table[{ {2, 1, 4}, 5 }] = 124LL;
    static_table[{ {2, 1, 4}, 6 }] = 1200LL;
    static_table[{ {2, 1, 4}, 7 }] = 12181LL;
    static_table[{ {2, 1, 4}, 8 }] = 117769LL;
    static_table[{ {2, 1, 5}, 5 }] = 124LL;
    static_table[{ {2, 1, 5}, 6 }] = 1205LL;
    static_table[{ {2, 1, 5}, 7 }] = 12857LL;
    static_table[{ {2, 1, 5}, 8 }] = 140592LL;
    static_table[{ {2, 2, 1}, 5 }] = 124LL;
    static_table[{ {2, 2, 1}, 6 }] = 1201LL;
    static_table[{ {2, 2, 1}, 7 }] = 12392LL;
    static_table[{ {2, 2, 1}, 8 }] = 126600LL;
    static_table[{ {2, 2, 2}, 5 }] = 124LL;
    static_table[{ {2, 2, 2}, 6 }] = 1184LL;
    static_table[{ {2, 2, 2}, 7 }] = 11181LL;
    static_table[{ {2, 2, 2}, 8 }] = 92601LL;
    static_table[{ {2, 2, 3}, 5 }] = 124LL;
    static_table[{ {2, 2, 3}, 6 }] = 1175LL;
    static_table[{ {2, 2, 3}, 7 }] = 10459LL;
    static_table[{ {2, 2, 3}, 8 }] = 74492LL;
    static_table[{ {2, 2, 4}, 5 }] = 124LL;
    static_table[{ {2, 2, 4}, 6 }] = 1184LL;
    static_table[{ {2, 2, 4}, 7 }] = 11181LL;
    static_table[{ {2, 2, 4}, 8 }] = 92601LL;
    static_table[{ {2, 2, 5}, 5 }] = 124LL;
    static_table[{ {2, 2, 5}, 6 }] = 1201LL;
    static_table[{ {2, 2, 5}, 7 }] = 12392LL;
    static_table[{ {2, 2, 5}, 8 }] = 126600LL;
    static_table[{ {2, 3, 1}, 5 }] = 124LL;
    static_table[{ {2, 3, 1}, 6 }] = 1193LL;
    static_table[{ {2, 3, 1}, 7 }] = 11783LL;
    static_table[{ {2, 3, 1}, 8 }] = 107147LL;
    static_table[{ {2, 3, 2}, 5 }] = 124LL;
    static_table[{ {2, 3, 2}, 6 }] = 1175LL;
    static_table[{ {2, 3, 2}, 7 }] = 10459LL;
    static_table[{ {2, 3, 2}, 8 }] = 74492LL;
    static_table[{ {2, 3, 3}, 5 }] = 124LL;
    static_table[{ {2, 3, 3}, 6 }] = 1168LL;
    static_table[{ {2, 3, 3}, 7 }] = 10243LL;
    static_table[{ {2, 3, 3}, 8 }] = 71832LL;
    static_table[{ {2, 3, 4}, 5 }] = 124LL;
    static_table[{ {2, 3, 4}, 6 }] = 1175LL;
    static_table[{ {2, 3, 4}, 7 }] = 10459LL;
    static_table[{ {2, 3, 4}, 8 }] = 74492LL;
    static_table[{ {2, 3, 5}, 5 }] = 124LL;
    static_table[{ {2, 3, 5}, 6 }] = 1193LL;
    static_table[{ {2, 3, 5}, 7 }] = 11783LL;
    static_table[{ {2, 3, 5}, 8 }] = 107147LL;
    static_table[{ {2, 4, 1}, 5 }] = 124LL;
    static_table[{ {2, 4, 1}, 6 }] = 1198LL;
    static_table[{ {2, 4, 1}, 7 }] = 12053LL;
    static_table[{ {2, 4, 1}, 8 }] = 114674LL;
    static_table[{ {2, 4, 2}, 5 }] = 124LL;
    static_table[{ {2, 4, 2}, 6 }] = 1184LL;
    static_table[{ {2, 4, 2}, 7 }] = 11181LL;
    static_table[{ {2, 4, 2}, 8 }] = 92601LL;
    static_table[{ {2, 4, 3}, 5 }] = 124LL;
    static_table[{ {2, 4, 3}, 6 }] = 1175LL;
    static_table[{ {2, 4, 3}, 7 }] = 10459LL;
    static_table[{ {2, 4, 3}, 8 }] = 74492LL;
    static_table[{ {2, 4, 4}, 5 }] = 124LL;
    static_table[{ {2, 4, 4}, 6 }] = 1184LL;
    static_table[{ {2, 4, 4}, 7 }] = 11181LL;
    static_table[{ {2, 4, 4}, 8 }] = 92601LL;
    static_table[{ {2, 4, 5}, 5 }] = 124LL;
    static_table[{ {2, 4, 5}, 6 }] = 1201LL;
    static_table[{ {2, 4, 5}, 7 }] = 12392LL;
    static_table[{ {2, 4, 5}, 8 }] = 126600LL;
    static_table[{ {2, 5, 1}, 5 }] = 124LL;
    static_table[{ {2, 5, 1}, 6 }] = 1205LL;
    static_table[{ {2, 5, 1}, 7 }] = 12857LL;
    static_table[{ {2, 5, 1}, 8 }] = 140592LL;
    static_table[{ {2, 5, 2}, 5 }] = 124LL;
    static_table[{ {2, 5, 2}, 6 }] = 1200LL;
    static_table[{ {2, 5, 2}, 7 }] = 12181LL;
    static_table[{ {2, 5, 2}, 8 }] = 117769LL;
    static_table[{ {2, 5, 3}, 5 }] = 124LL;
    static_table[{ {2, 5, 3}, 6 }] = 1198LL;
    static_table[{ {2, 5, 3}, 7 }] = 12053LL;
    static_table[{ {2, 5, 3}, 8 }] = 114674LL;
    static_table[{ {2, 5, 4}, 5 }] = 124LL;
    static_table[{ {2, 5, 4}, 6 }] = 1206LL;
    static_table[{ {2, 5, 4}, 7 }] = 12759LL;
    static_table[{ {2, 5, 4}, 8 }] = 139178LL;
    static_table[{ {2, 5, 5}, 5 }] = 124LL;
    static_table[{ {2, 5, 5}, 6 }] = 1211LL;
    static_table[{ {2, 5, 5}, 7 }] = 13310LL;
    static_table[{ {2, 5, 5}, 8 }] = 153074LL;
    static_table[{ {3, 1, 1}, 5 }] = 124LL;
    static_table[{ {3, 1, 1}, 6 }] = 1205LL;
    static_table[{ {3, 1, 1}, 7 }] = 12857LL;
    static_table[{ {3, 1, 1}, 8 }] = 140592LL;
    static_table[{ {3, 1, 2}, 5 }] = 124LL;
    static_table[{ {3, 1, 2}, 6 }] = 1193LL;
    static_table[{ {3, 1, 2}, 7 }] = 11783LL;
    static_table[{ {3, 1, 2}, 8 }] = 107147LL;
    static_table[{ {3, 1, 3}, 5 }] = 124LL;
    static_table[{ {3, 1, 3}, 6 }] = 1184LL;
    static_table[{ {3, 1, 3}, 7 }] = 11181LL;
    static_table[{ {3, 1, 3}, 8 }] = 92601LL;
    static_table[{ {3, 1, 4}, 5 }] = 124LL;
    static_table[{ {3, 1, 4}, 6 }] = 1193LL;
    static_table[{ {3, 1, 4}, 7 }] = 11786LL;
    static_table[{ {3, 1, 4}, 8 }] = 107275LL;
    static_table[{ {3, 1, 5}, 5 }] = 124LL;
    static_table[{ {3, 1, 5}, 6 }] = 1205LL;
    static_table[{ {3, 1, 5}, 7 }] = 12857LL;
    static_table[{ {3, 1, 5}, 8 }] = 140592LL;
    static_table[{ {3, 2, 1}, 5 }] = 124LL;
    static_table[{ {3, 2, 1}, 6 }] = 1198LL;
    static_table[{ {3, 2, 1}, 7 }] = 12053LL;
    static_table[{ {3, 2, 1}, 8 }] = 114674LL;
    static_table[{ {3, 2, 2}, 5 }] = 124LL;
    static_table[{ {3, 2, 2}, 6 }] = 1175LL;
    static_table[{ {3, 2, 2}, 7 }] = 10459LL;
    static_table[{ {3, 2, 2}, 8 }] = 74492LL;
    static_table[{ {3, 2, 3}, 5 }] = 124LL;
    static_table[{ {3, 2, 3}, 6 }] = 1162LL;
    static_table[{ {3, 2, 3}, 7 }] = 9706LL;
    static_table[{ {3, 2, 3}, 8 }] = 62719LL;
    static_table[{ {3, 2, 4}, 5 }] = 124LL;
    static_table[{ {3, 2, 4}, 6 }] = 1175LL;
    static_table[{ {3, 2, 4}, 7 }] = 10459LL;
    static_table[{ {3, 2, 4}, 8 }] = 74492LL;
    static_table[{ {3, 2, 5}, 5 }] = 124LL;
    static_table[{ {3, 2, 5}, 6 }] = 1198LL;
    static_table[{ {3, 2, 5}, 7 }] = 12053LL;
    static_table[{ {3, 2, 5}, 8 }] = 114674LL;
    static_table[{ {3, 3, 1}, 5 }] = 124LL;
    static_table[{ {3, 3, 1}, 6 }] = 1190LL;
    static_table[{ {3, 3, 1}, 7 }] = 11849LL;
    static_table[{ {3, 3, 1}, 8 }] = 114128LL;
    static_table[{ {3, 3, 2}, 5 }] = 124LL;
    static_table[{ {3, 3, 2}, 6 }] = 1168LL;
    static_table[{ {3, 3, 2}, 7 }] = 10243LL;
    static_table[{ {3, 3, 2}, 8 }] = 71832LL;
    static_table[{ {3, 3, 3}, 5 }] = 124LL;
    static_table[{ {3, 3, 3}, 6 }] = 1144LL;
    static_table[{ {3, 3, 3}, 7 }] = 9265LL;
    static_table[{ {3, 3, 3}, 8 }] = 59800LL;
    static_table[{ {3, 3, 4}, 5 }] = 124LL;
    static_table[{ {3, 3, 4}, 6 }] = 1168LL;
    static_table[{ {3, 3, 4}, 7 }] = 10243LL;
    static_table[{ {3, 3, 4}, 8 }] = 71832LL;
    static_table[{ {3, 3, 5}, 5 }] = 124LL;
    static_table[{ {3, 3, 5}, 6 }] = 1190LL;
    static_table[{ {3, 3, 5}, 7 }] = 11849LL;
    static_table[{ {3, 3, 5}, 8 }] = 114128LL;
    static_table[{ {3, 4, 1}, 5 }] = 124LL;
    static_table[{ {3, 4, 1}, 6 }] = 1199LL;
    static_table[{ {3, 4, 1}, 7 }] = 12392LL;
    static_table[{ {3, 4, 1}, 8 }] = 126722LL;
    static_table[{ {3, 4, 2}, 5 }] = 124LL;
    static_table[{ {3, 4, 2}, 6 }] = 1175LL;
    static_table[{ {3, 4, 2}, 7 }] = 10459LL;
    static_table[{ {3, 4, 2}, 8 }] = 74492LL;
    static_table[{ {3, 4, 3}, 5 }] = 124LL;
    static_table[{ {3, 4, 3}, 6 }] = 1168LL;
    static_table[{ {3, 4, 3}, 7 }] = 10243LL;
    static_table[{ {3, 4, 3}, 8 }] = 71832LL;
    static_table[{ {3, 4, 4}, 5 }] = 124LL;
    static_table[{ {3, 4, 4}, 6 }] = 1175LL;
    static_table[{ {3, 4, 4}, 7 }] = 10459LL;
    static_table[{ {3, 4, 4}, 8 }] = 74492LL;
    static_table[{ {3, 4, 5}, 5 }] = 124LL;
    static_table[{ {3, 4, 5}, 6 }] = 1199LL;
    static_table[{ {3, 4, 5}, 7 }] = 12392LL;
    static_table[{ {3, 4, 5}, 8 }] = 126722LL;
    static_table[{ {3, 5, 1}, 5 }] = 124LL;
    static_table[{ {3, 5, 1}, 6 }] = 1207LL;
    static_table[{ {3, 5, 1}, 7 }] = 12903LL;
    static_table[{ {3, 5, 1}, 8 }] = 142345LL;
    static_table[{ {3, 5, 2}, 5 }] = 124LL;
    static_table[{ {3, 5, 2}, 6 }] = 1193LL;
    static_table[{ {3, 5, 2}, 7 }] = 11786LL;
    static_table[{ {3, 5, 2}, 8 }] = 107275LL;
    static_table[{ {3, 5, 3}, 5 }] = 124LL;
    static_table[{ {3, 5, 3}, 6 }] = 1184LL;
    static_table[{ {3, 5, 3}, 7 }] = 11181LL;
    static_table[{ {3, 5, 3}, 8 }] = 92601LL;
    static_table[{ {3, 5, 4}, 5 }] = 124LL;
    static_table[{ {3, 5, 4}, 6 }] = 1193LL;
    static_table[{ {3, 5, 4}, 7 }] = 11783LL;
    static_table[{ {3, 5, 4}, 8 }] = 107147LL;
    static_table[{ {3, 5, 5}, 5 }] = 124LL;
    static_table[{ {3, 5, 5}, 6 }] = 1205LL;
    static_table[{ {3, 5, 5}, 7 }] = 12857LL;
    static_table[{ {3, 5, 5}, 8 }] = 140592LL;
    static_table[{ {4, 1, 1}, 5 }] = 124LL;
    static_table[{ {4, 1, 1}, 6 }] = 1211LL;
    static_table[{ {4, 1, 1}, 7 }] = 13311LL;
    static_table[{ {4, 1, 1}, 8 }] = 153137LL;
    static_table[{ {4, 1, 2}, 5 }] = 124LL;
    static_table[{ {4, 1, 2}, 6 }] = 1198LL;
    static_table[{ {4, 1, 2}, 7 }] = 12053LL;
    static_table[{ {4, 1, 2}, 8 }] = 114674LL;
    static_table[{ {4, 1, 3}, 5 }] = 124LL;
    static_table[{ {4, 1, 3}, 6 }] = 1189LL;
    static_table[{ {4, 1, 3}, 7 }] = 11311LL;
    static_table[{ {4, 1, 3}, 8 }] = 97136LL;
    static_table[{ {4, 1, 4}, 5 }] = 124LL;
    static_table[{ {4, 1, 4}, 6 }] = 1198LL;
    static_table[{ {4, 1, 4}, 7 }] = 12053LL;
    static_table[{ {4, 1, 4}, 8 }] = 114674LL;
    static_table[{ {4, 1, 5}, 5 }] = 124LL;
    static_table[{ {4, 1, 5}, 6 }] = 1211LL;
    static_table[{ {4, 1, 5}, 7 }] = 13311LL;
    static_table[{ {4, 1, 5}, 8 }] = 153137LL;
    static_table[{ {4, 2, 1}, 5 }] = 124LL;
    static_table[{ {4, 2, 1}, 6 }] = 1199LL;
    static_table[{ {4, 2, 1}, 7 }] = 12392LL;
    static_table[{ {4, 2, 1}, 8 }] = 126722LL;
    static_table[{ {4, 2, 2}, 5 }] = 124LL;
    static_table[{ {4, 2, 2}, 6 }] = 1175LL;
    static_table[{ {4, 2, 2}, 7 }] = 10459LL;
    static_table[{ {4, 2, 2}, 8 }] = 74492LL;
    static_table[{ {4, 2, 3}, 5 }] = 124LL;
    static_table[{ {4, 2, 3}, 6 }] = 1168LL;
    static_table[{ {4, 2, 3}, 7 }] = 10243LL;
    static_table[{ {4, 2, 3}, 8 }] = 71832LL;
    static_table[{ {4, 2, 4}, 5 }] = 124LL;
    static_table[{ {4, 2, 4}, 6 }] = 1175LL;
    static_table[{ {4, 2, 4}, 7 }] = 10459LL;
    static_table[{ {4, 2, 4}, 8 }] = 74492LL;
    static_table[{ {4, 2, 5}, 5 }] = 124LL;
    static_table[{ {4, 2, 5}, 6 }] = 1199LL;
    static_table[{ {4, 2, 5}, 7 }] = 12392LL;
    static_table[{ {4, 2, 5}, 8 }] = 126722LL;
    static_table[{ {4, 3, 1}, 5 }] = 124LL;
    static_table[{ {4, 3, 1}, 6 }] = 1209LL;
    static_table[{ {4, 3, 1}, 7 }] = 12954LL;
    static_table[{ {4, 3, 1}, 8 }] = 140555LL;
    static_table[{ {4, 3, 2}, 5 }] = 124LL;
    static_table[{ {4, 3, 2}, 6 }] = 1204LL;
    static_table[{ {4, 3, 2}, 7 }] = 12523LL;
    static_table[{ {4, 3, 2}, 8 }] = 125347LL;
    static_table[{ {4, 3, 3}, 5 }] = 124LL;
    static_table[{ {4, 3, 3}, 6 }] = 1190LL;
    static_table[{ {4, 3, 3}, 7 }] = 11849LL;
    static_table[{ {4, 3, 3}, 8 }] = 114128LL;
    static_table[{ {4, 3, 4}, 5 }] = 124LL;
    static_table[{ {4, 3, 4}, 6 }] = 1193LL;
    static_table[{ {4, 3, 4}, 7 }] = 11786LL;
    static_table[{ {4, 3, 4}, 8 }] = 107275LL;
    static_table[{ {4, 3, 5}, 5 }] = 124LL;
    static_table[{ {4, 3, 5}, 6 }] = 1216LL;
    static_table[{ {4, 3, 5}, 7 }] = 13588LL;
    static_table[{ {4, 3, 5}, 8 }] = 162521LL;
    static_table[{ {4, 4, 1}, 5 }] = 124LL;
    static_table[{ {4, 4, 1}, 6 }] = 1211LL;
    static_table[{ {4, 4, 1}, 7 }] = 13311LL;
    static_table[{ {4, 4, 1}, 8 }] = 153137LL;
    static_table[{ {4, 4, 2}, 5 }] = 124LL;
    static_table[{ {4, 4, 2}, 6 }] = 1194LL;
    static_table[{ {4, 4, 2}, 7 }] = 12134LL;
    static_table[{ {4, 4, 2}, 8 }] = 121074LL;
    static_table[{ {4, 4, 3}, 5 }] = 124LL;
    static_table[{ {4, 4, 3}, 6 }] = 1199LL;
    static_table[{ {4, 4, 3}, 7 }] = 12392LL;
    static_table[{ {4, 4, 3}, 8 }] = 126722LL;
    static_table[{ {4, 4, 4}, 5 }] = 124LL;
    static_table[{ {4, 4, 4}, 6 }] = 1168LL;
    static_table[{ {4, 4, 4}, 7 }] = 12118LL;
    static_table[{ {4, 4, 4}, 8 }] = 135046LL;
    static_table[{ {4, 4, 5}, 5 }] = 124LL;
    static_table[{ {4, 4, 5}, 6 }] = 1205LL;
    static_table[{ {4, 4, 5}, 7 }] = 12857LL;
    static_table[{ {4, 4, 5}, 8 }] = 140592LL;
    static_table[{ {4, 5, 1}, 5 }] = 124LL;
    static_table[{ {4, 5, 1}, 6 }] = 1219LL;
    static_table[{ {4, 5, 1}, 7 }] = 13755LL;
    static_table[{ {4, 5, 1}, 8 }] = 167123LL;
    static_table[{ {4, 5, 2}, 5 }] = 124LL;
    static_table[{ {4, 5, 2}, 6 }] = 1212LL;
    static_table[{ {4, 5, 2}, 7 }] = 13213LL;
    static_table[{ {4, 5, 2}, 8 }] = 150493LL;
    static_table[{ {4, 5, 3}, 5 }] = 124LL;
    static_table[{ {4, 5, 3}, 6 }] = 1209LL;
    static_table[{ {4, 5, 3}, 7 }] = 12954LL;
    static_table[{ {4, 5, 3}, 8 }] = 140555LL;
    static_table[{ {4, 5, 4}, 5 }] = 124LL;
    static_table[{ {4, 5, 4}, 6 }] = 1211LL;
    static_table[{ {4, 5, 4}, 7 }] = 13311LL;
    static_table[{ {4, 5, 4}, 8 }] = 153137LL;
    static_table[{ {4, 5, 5}, 5 }] = 124LL;
    static_table[{ {4, 5, 5}, 6 }] = 1212LL;
    static_table[{ {4, 5, 5}, 7 }] = 13213LL;
    static_table[{ {4, 5, 5}, 8 }] = 150493LL;
    static_table[{ {5, 1, 1}, 5 }] = 124LL;
    static_table[{ {5, 1, 1}, 6 }] = 1216LL;
    static_table[{ {5, 1, 1}, 7 }] = 13624LL;
    static_table[{ {5, 1, 1}, 8 }] = 165376LL;
    static_table[{ {5, 1, 2}, 5 }] = 124LL;
    static_table[{ {5, 1, 2}, 6 }] = 1209LL;
    static_table[{ {5, 1, 2}, 7 }] = 12954LL;
    static_table[{ {5, 1, 2}, 8 }] = 140555LL;
    static_table[{ {5, 1, 3}, 5 }] = 124LL;
    static_table[{ {5, 1, 3}, 6 }] = 1204LL;
    static_table[{ {5, 1, 3}, 7 }] = 12523LL;
    static_table[{ {5, 1, 3}, 8 }] = 125347LL;
    static_table[{ {5, 1, 4}, 5 }] = 124LL;
    static_table[{ {5, 1, 4}, 6 }] = 1211LL;
    static_table[{ {5, 1, 4}, 7 }] = 13311LL;
    static_table[{ {5, 1, 4}, 8 }] = 153137LL;
    static_table[{ {5, 1, 5}, 5 }] = 124LL;
    static_table[{ {5, 1, 5}, 6 }] = 1212LL;
    static_table[{ {5, 1, 5}, 7 }] = 13213LL;
    static_table[{ {5, 1, 5}, 8 }] = 150493LL;
    static_table[{ {5, 2, 1}, 5 }] = 124LL;
    static_table[{ {5, 2, 1}, 6 }] = 1212LL;
    static_table[{ {5, 2, 1}, 7 }] = 13213LL;
    static_table[{ {5, 2, 1}, 8 }] = 150493LL;
    static_table[{ {5, 2, 2}, 5 }] = 124LL;
    static_table[{ {5, 2, 2}, 6 }] = 1205LL;
    static_table[{ {5, 2, 2}, 7 }] = 12857LL;
    static_table[{ {5, 2, 2}, 8 }] = 140592LL;
    static_table[{ {5, 2, 3}, 5 }] = 124LL;
    static_table[{ {5, 2, 3}, 6 }] = 1199LL;
    static_table[{ {5, 2, 3}, 7 }] = 12392LL;
    static_table[{ {5, 2, 3}, 8 }] = 126722LL;
    static_table[{ {5, 2, 4}, 5 }] = 124LL;
    static_table[{ {5, 2, 4}, 6 }] = 1209LL;
    static_table[{ {5, 2, 4}, 7 }] = 12954LL;
    static_table[{ {5, 2, 4}, 8 }] = 140555LL;
    static_table[{ {5, 2, 5}, 5 }] = 124LL;
    static_table[{ {5, 2, 5}, 6 }] = 1219LL;
    static_table[{ {5, 2, 5}, 7 }] = 13755LL;
    static_table[{ {5, 2, 5}, 8 }] = 167123LL;
    static_table[{ {5, 3, 1}, 5 }] = 124LL;
    static_table[{ {5, 3, 1}, 6 }] = 1204LL;
    static_table[{ {5, 3, 1}, 7 }] = 12523LL;
    static_table[{ {5, 3, 1}, 8 }] = 125347LL;
    static_table[{ {5, 3, 2}, 5 }] = 124LL;
    static_table[{ {5, 3, 2}, 6 }] = 1193LL;
    static_table[{ {5, 3, 2}, 7 }] = 11786LL;
    static_table[{ {5, 3, 2}, 8 }] = 107275LL;
    static_table[{ {5, 3, 3}, 5 }] = 124LL;
    static_table[{ {5, 3, 3}, 6 }] = 1190LL;
    static_table[{ {5, 3, 3}, 7 }] = 11849LL;
    static_table[{ {5, 3, 3}, 8 }] = 114128LL;
    static_table[{ {5, 3, 4}, 5 }] = 124LL;
    static_table[{ {5, 3, 4}, 6 }] = 1204LL;
    static_table[{ {5, 3, 4}, 7 }] = 12523LL;
    static_table[{ {5, 3, 4}, 8 }] = 125347LL;
    static_table[{ {5, 3, 5}, 5 }] = 124LL;
    static_table[{ {5, 3, 5}, 6 }] = 1209LL;
    static_table[{ {5, 3, 5}, 7 }] = 12954LL;
    static_table[{ {5, 3, 5}, 8 }] = 140555LL;
    static_table[{ {5, 4, 1}, 5 }] = 124LL;
    static_table[{ {5, 4, 1}, 6 }] = 1211LL;
    static_table[{ {5, 4, 1}, 7 }] = 13311LL;
    static_table[{ {5, 4, 1}, 8 }] = 153137LL;
    static_table[{ {5, 4, 2}, 5 }] = 124LL;
    static_table[{ {5, 4, 2}, 6 }] = 1205LL;
    static_table[{ {5, 4, 2}, 7 }] = 12857LL;
    static_table[{ {5, 4, 2}, 8 }] = 140592LL;
    static_table[{ {5, 4, 3}, 5 }] = 124LL;
    static_table[{ {5, 4, 3}, 6 }] = 1204LL;
    static_table[{ {5, 4, 3}, 7 }] = 12523LL;
    static_table[{ {5, 4, 3}, 8 }] = 125347LL;
    static_table[{ {5, 4, 4}, 5 }] = 124LL;
    static_table[{ {5, 4, 4}, 6 }] = 1212LL;
    static_table[{ {5, 4, 4}, 7 }] = 13213LL;
    static_table[{ {5, 4, 4}, 8 }] = 150493LL;
    static_table[{ {5, 4, 5}, 5 }] = 124LL;
    static_table[{ {5, 4, 5}, 6 }] = 1212LL;
    static_table[{ {5, 4, 5}, 7 }] = 13213LL;
    static_table[{ {5, 4, 5}, 8 }] = 150493LL;
    static_table[{ {5, 5, 1}, 5 }] = 124LL;
    static_table[{ {5, 5, 1}, 6 }] = 1216LL;
    static_table[{ {5, 5, 1}, 7 }] = 13624LL;
    static_table[{ {5, 5, 1}, 8 }] = 165376LL;
    static_table[{ {5, 5, 2}, 5 }] = 124LL;
    static_table[{ {5, 5, 2}, 6 }] = 1212LL;
    static_table[{ {5, 5, 2}, 7 }] = 13213LL;
    static_table[{ {5, 5, 2}, 8 }] = 150493LL;
    static_table[{ {5, 5, 3}, 5 }] = 124LL;
    static_table[{ {5, 5, 3}, 6 }] = 1209LL;
    static_table[{ {5, 5, 3}, 7 }] = 12954LL;
    static_table[{ {5, 5, 3}, 8 }] = 140555LL;
    static_table[{ {5, 5, 4}, 5 }] = 124LL;
    static_table[{ {5, 5, 4}, 6 }] = 1212LL;
    static_table[{ {5, 5, 4}, 7 }] = 13213LL;
    static_table[{ {5, 5, 4}, 8 }] = 150493LL;
    static_table[{ {5, 5, 5}, 5 }] = 124LL;
    static_table[{ {5, 5, 5}, 6 }] = 1216LL;
    static_table[{ {5, 5, 5}, 7 }] = 13624LL;
    static_table[{ {5, 5, 5}, 8 }] = 165376LL;
    static_table[{ {}, 9 }] = 0LL;
    static_table[{ {}, 10 }] = 0LL;
    static_table[{ {}, 11 }] = 0LL;
    static_table[{ {}, 12 }] = 0LL;
    static_table[{ {1}, 9 }] = 128LL;
    static_table[{ {1}, 10 }] = 256LL;
    static_table[{ {1}, 11 }] = 512LL;
    static_table[{ {1}, 12 }] = 1024LL;
    static_table[{ {2}, 9 }] = 0LL;
    static_table[{ {2}, 10 }] = 0LL;
    static_table[{ {2}, 11 }] = 0LL;
    static_table[{ {2}, 12 }] = 0LL;
    static_table[{ {3}, 9 }] = 128LL;
    static_table[{ {3}, 10 }] = 256LL;
    static_table[{ {3}, 11 }] = 512LL;
    static_table[{ {3}, 12 }] = 1024LL;
    static_table[{ {1, 1}, 9 }] = 360693LL;
    static_table[{ {1, 1}, 10 }] = 3622239LL;
    static_table[{ {1, 1}, 11 }] = 39897117LL;
    static_table[{ {1, 1}, 12 }] = 478942551LL;
    static_table[{ {1, 2}, 9 }] = 104486LL;
    static_table[{ {1, 2}, 10 }] = 634516LL;
    static_table[{ {1, 3}, 9 }] = 47763LL;
    static_table[{ {1, 3}, 10 }] = 261365LL;
    static_table[{ {1, 4}, 9 }] = 27752LL;
    static_table[{ {1, 4}, 10 }] = 92905LL;
    static_table[{ {2, 1}, 9 }] = 123806LL;
    static_table[{ {2, 1}, 10 }] = 772367LL;
    static_table[{ {2, 2}, 9 }] = 191443LL;
    static_table[{ {2, 2}, 10 }] = 1886243LL;
    static_table[{ {2, 3}, 9 }] = 3784LL;
    static_table[{ {2, 3}, 10 }] = 7992LL;
    static_table[{ {2, 4}, 9 }] = 47763LL;
    static_table[{ {2, 4}, 10 }] = 261365LL;
    static_table[{ {3, 1}, 9 }] = 55529LL;
    static_table[{ {3, 1}, 10 }] = 263556LL;
    static_table[{ {3, 2}, 9 }] = 10969LL;
    static_table[{ {3, 2}, 10 }] = 32764LL;
    static_table[{ {3, 3}, 9 }] = 191443LL;
    static_table[{ {3, 3}, 10 }] = 1886243LL;
    static_table[{ {3, 4}, 9 }] = 104486LL;
    static_table[{ {3, 4}, 10 }] = 634516LL;
    static_table[{ {4, 1}, 9 }] = 87745LL;
    static_table[{ {4, 1}, 10 }] = 487777LL;
    static_table[{ {4, 2}, 9 }] = 55529LL;
    static_table[{ {4, 2}, 10 }] = 263556LL;
    static_table[{ {4, 3}, 9 }] = 123806LL;
    static_table[{ {4, 3}, 10 }] = 772367LL;
    static_table[{ {4, 4}, 9 }] = 360693LL;
    static_table[{ {4, 4}, 10 }] = 3622239LL;
    static_table[{ {1, 1, 1}, 9 }] = 2133184LL;
    static_table[{ {1, 1, 1}, 10 }] = 29095936LL;
    static_table[{ {1, 1, 2}, 9 }] = 1955701LL;
    static_table[{ {1, 1, 2}, 10 }] = 24032924LL;
    static_table[{ {1, 1, 3}, 9 }] = 1521031LL;
    static_table[{ {1, 1, 3}, 10 }] = 16378506LL;
    static_table[{ {1, 1, 4}, 9 }] = 1809018LL;
    static_table[{ {1, 1, 4}, 10 }] = 20716642LL;
    static_table[{ {1, 1, 5}, 9 }] = 1912965LL;
    static_table[{ {1, 2, 1}, 9 }] = 1585723LL;
    static_table[{ {1, 2, 2}, 9 }] = 1552560LL;
    static_table[{ {1, 2, 3}, 9 }] = 1612950LL;
    static_table[{ {1, 2, 4}, 9 }] = 1442599LL;
    static_table[{ {1, 2, 5}, 9 }] = 1487151LL;
    static_table[{ {1, 3, 1}, 9 }] = 1295553LL;
    static_table[{ {1, 3, 2}, 9 }] = 1752014LL;
    static_table[{ {1, 3, 3}, 9 }] = 1441781LL;
    static_table[{ {1, 3, 4}, 9 }] = 1268280LL;
    static_table[{ {1, 3, 5}, 9 }] = 1346623LL;
    static_table[{ {1, 4, 1}, 9 }] = 1290338LL;
    static_table[{ {1, 4, 2}, 9 }] = 1540329LL;
    static_table[{ {1, 4, 3}, 9 }] = 1336449LL;
    static_table[{ {1, 4, 4}, 9 }] = 1318768LL;
    static_table[{ {1, 4, 5}, 9 }] = 1487151LL;
    static_table[{ {1, 5, 1}, 9 }] = 1636386LL;
    static_table[{ {1, 5, 2}, 9 }] = 2106008LL;
    static_table[{ {1, 5, 3}, 9 }] = 1885916LL;
    static_table[{ {1, 5, 4}, 9 }] = 1764811LL;
    static_table[{ {1, 5, 5}, 9 }] = 1912965LL;
    static_table[{ {2, 1, 1}, 9 }] = 1737509LL;
    static_table[{ {2, 1, 2}, 9 }] = 1458772LL;
    static_table[{ {2, 1, 3}, 9 }] = 1979588LL;
    static_table[{ {2, 1, 4}, 9 }] = 1761515LL;
    static_table[{ {2, 1, 5}, 9 }] = 1764811LL;
    static_table[{ {2, 2, 1}, 9 }] = 1879244LL;
    static_table[{ {2, 2, 2}, 9 }] = 1610650LL;
    static_table[{ {2, 2, 3}, 9 }] = 1059237LL;
    static_table[{ {2, 2, 4}, 9 }] = 1210341LL;
    static_table[{ {2, 2, 5}, 9 }] = 1318768LL;
    static_table[{ {2, 3, 1}, 9 }] = 1833335LL;
    static_table[{ {2, 3, 2}, 9 }] = 938690LL;
    static_table[{ {2, 3, 3}, 9 }] = 1259051LL;
    static_table[{ {2, 3, 4}, 9 }] = 991767LL;
    static_table[{ {2, 3, 5}, 9 }] = 1268280LL;
    static_table[{ {2, 4, 1}, 9 }] = 1687147LL;
    static_table[{ {2, 4, 2}, 9 }] = 936876LL;
    static_table[{ {2, 4, 3}, 9 }] = 1084334LL;
    static_table[{ {2, 4, 4}, 9 }] = 1210341LL;
    static_table[{ {2, 4, 5}, 9 }] = 1442599LL;
    static_table[{ {2, 5, 1}, 9 }] = 2174604LL;
    static_table[{ {2, 5, 2}, 9 }] = 1192736LL;
    static_table[{ {2, 5, 3}, 9 }] = 1573805LL;
    static_table[{ {2, 5, 4}, 9 }] = 1761515LL;
    static_table[{ {2, 5, 5}, 9 }] = 1809018LL;
    static_table[{ {3, 1, 1}, 9 }] = 1664213LL;
    static_table[{ {3, 1, 2}, 9 }] = 2014996LL;
    static_table[{ {3, 1, 3}, 9 }] = 1230250LL;
    static_table[{ {3, 1, 4}, 9 }] = 1573805LL;
    static_table[{ {3, 1, 5}, 9 }] = 1885916LL;
    static_table[{ {3, 2, 1}, 9 }] = 1710223LL;
    static_table[{ {3, 2, 2}, 9 }] = 1305780LL;
    static_table[{ {3, 2, 3}, 9 }] = 919674LL;
    static_table[{ {3, 2, 4}, 9 }] = 1084334LL;
    static_table[{ {3, 2, 5}, 9 }] = 1336449LL;
    static_table[{ {3, 3, 1}, 9 }] = 1585128LL;
    static_table[{ {3, 3, 2}, 9 }] = 1083268LL;
    static_table[{ {3, 3, 3}, 9 }] = 1492704LL;
    static_table[{ {3, 3, 4}, 9 }] = 1259051LL;
    static_table[{ {3, 3, 5}, 9 }] = 1441781LL;
    static_table[{ {3, 4, 1}, 9 }] = 1518775LL;
    static_table[{ {3, 4, 2}, 9 }] = 1147547LL;
    static_table[{ {3, 4, 3}, 9 }] = 919674LL;
    static_table[{ {3, 4, 4}, 9 }] = 1059237LL;
    static_table[{ {3, 4, 5}, 9 }] = 1612950LL;
    static_table[{ {3, 5, 1}, 9 }] = 1845543LL;
    static_table[{ {3, 5, 2}, 9 }] = 1570219LL;
    static_table[{ {3, 5, 3}, 9 }] = 1230250LL;
    static_table[{ {3, 5, 4}, 9 }] = 1979588LL;
    static_table[{ {3, 5, 5}, 9 }] = 1521031LL;
    static_table[{ {4, 1, 1}, 9 }] = 1494562LL;
    static_table[{ {4, 1, 2}, 9 }] = 1739368LL;
    static_table[{ {4, 1, 3}, 9 }] = 1570219LL;
    static_table[{ {4, 1, 4}, 9 }] = 1192736LL;
    static_table[{ {4, 1, 5}, 9 }] = 2106008LL;
    static_table[{ {4, 2, 1}, 9 }] = 1601020LL;
    static_table[{ {4, 2, 2}, 9 }] = 1207311LL;
    static_table[{ {4, 2, 3}, 9 }] = 1147547LL;
    static_table[{ {4, 2, 4}, 9 }] = 936876LL;
    static_table[{ {4, 2, 5}, 9 }] = 1540329LL;
    static_table[{ {4, 3, 1}, 9 }] = 1507559LL;
    static_table[{ {4, 3, 2}, 9 }] = 1193960LL;
    static_table[{ {4, 3, 3}, 9 }] = 1083268LL;
    static_table[{ {4, 3, 4}, 9 }] = 938690LL;
    static_table[{ {4, 3, 5}, 9 }] = 1752014LL;
    static_table[{ {4, 4, 1}, 9 }] = 1793647LL;
    static_table[{ {4, 4, 2}, 9 }] = 1207311LL;
    static_table[{ {4, 4, 3}, 9 }] = 1305780LL;
    static_table[{ {4, 4, 4}, 9 }] = 1610650LL;
    static_table[{ {4, 4, 5}, 9 }] = 1552560LL;
    static_table[{ {4, 5, 1}, 9 }] = 2106008LL;
    static_table[{ {4, 5, 2}, 9 }] = 1739368LL;
    static_table[{ {4, 5, 3}, 9 }] = 2014996LL;
    static_table[{ {4, 5, 4}, 9 }] = 1458772LL;
    static_table[{ {4, 5, 5}, 9 }] = 1955701LL;
    static_table[{ {5, 1, 1}, 9 }] = 2021415LL;
    static_table[{ {5, 1, 2}, 9 }] = 2106008LL;
    static_table[{ {5, 1, 3}, 9 }] = 1845543LL;
    static_table[{ {5, 1, 4}, 9 }] = 2174604LL;
    static_table[{ {5, 1, 5}, 9 }] = 1636386LL;
    static_table[{ {5, 2, 1}, 9 }] = 1857553LL;
    static_table[{ {5, 2, 2}, 9 }] = 1793647LL;
    static_table[{ {5, 2, 3}, 9 }] = 1518775LL;
    static_table[{ {5, 2, 4}, 9 }] = 1687147LL;
    static_table[{ {5, 2, 5}, 9 }] = 1290338LL;
    static_table[{ {5, 3, 1}, 9 }] = 1845543LL;
    static_table[{ {5, 3, 2}, 9 }] = 1507559LL;
    static_table[{ {5, 3, 3}, 9 }] = 1585128LL;
    static_table[{ {5, 3, 4}, 9 }] = 1833335LL;
    static_table[{ {5, 3, 5}, 9 }] = 1295553LL;
    static_table[{ {5, 4, 1}, 9 }] = 1857553LL;
    static_table[{ {5, 4, 2}, 9 }] = 1601020LL;
    static_table[{ {5, 4, 3}, 9 }] = 1710223LL;
    static_table[{ {5, 4, 4}, 9 }] = 1879244LL;
    static_table[{ {5, 4, 5}, 9 }] = 1585723LL;
    static_table[{ {5, 5, 1}, 9 }] = 2021415LL;
    static_table[{ {5, 5, 2}, 9 }] = 1494562LL;
    static_table[{ {5, 5, 3}, 9 }] = 1664213LL;
    static_table[{ {5, 5, 4}, 9 }] = 1737509LL;
    static_table[{ {5, 5, 5}, 9 }] = 2133184LL;
    static_table[{ {1, 1, 1, 1}, 9 }] = 4094995LL;
    static_table[{ {1, 1, 1, 2}, 9 }] = 3240939LL;
    static_table[{ {1, 1, 1, 3}, 9 }] = 2936040LL;
    static_table[{ {1, 1, 1, 4}, 9 }] = 2989208LL;
    static_table[{ {1, 1, 1, 5}, 9 }] = 3382942LL;
    static_table[{ {1, 1, 1, 6}, 9 }] = 3428147LL;
    static_table[{ {1, 1, 2, 1}, 9 }] = 3420680LL;
    static_table[{ {1, 1, 2, 2}, 9 }] = 3484901LL;
    static_table[{ {1, 1, 2, 3}, 9 }] = 3203840LL;
    static_table[{ {1, 1, 2, 4}, 9 }] = 3033842LL;
    static_table[{ {1, 1, 2, 5}, 9 }] = 3386991LL;
    static_table[{ {1, 1, 2, 6}, 9 }] = 3442676LL;
    static_table[{ {1, 1, 3, 1}, 9 }] = 3205691LL;
    static_table[{ {1, 1, 3, 2}, 9 }] = 3188032LL;
    static_table[{ {1, 1, 3, 3}, 9 }] = 3222023LL;
    static_table[{ {1, 1, 3, 4}, 9 }] = 2889258LL;
    static_table[{ {1, 1, 3, 5}, 9 }] = 3101247LL;
    static_table[{ {1, 1, 3, 6}, 9 }] = 3172964LL;
    static_table[{ {1, 1, 4, 1}, 9 }] = 3357046LL;
    static_table[{ {1, 1, 4, 2}, 9 }] = 3026272LL;
    static_table[{ {1, 1, 4, 3}, 9 }] = 2899309LL;
    static_table[{ {1, 1, 4, 4}, 9 }] = 3402309LL;
    static_table[{ {1, 1, 4, 5}, 9 }] = 3362691LL;
    static_table[{ {1, 1, 4, 6}, 9 }] = 3422875LL;
    static_table[{ {1, 1, 5, 1}, 9 }] = 3411055LL;
    static_table[{ {1, 1, 5, 2}, 9 }] = 3442870LL;
}
// 1. Add this helper function ABOVE your main()
void decode_prufer(long long idx, int t, vector<int>& res) {
    for (int i = 0; i < t - 2; ++i) {
        res[i] = (idx % t) + 1;
        idx /= t;
    }
}

// 2. Replace your entire main() with this:
int main() {
    // --- Fast I/O ---
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    load_table();

    int t = 12;
    // Cayley's Formula: t^(t-2)
    long long total_trees = (t == 2) ? 1 : 1;
    if (t > 2) {
        for(int i = 0; i < t - 2; ++i) total_trees *= t;
    }

    // This map stores results found during the current execution
    map<pair<vector<int>, int>, long long> memo_results;

    cout << "Target t = " << t << " | Total Trees to check: " << total_trees << endl;
    cout << "--------------------------------------------------------" << endl;

    for (int len = 0; len <= t-2; ++len) { 
        auto P_seqs = all_prufer_sequences(len + 2); 
        for (const auto& P : P_seqs) {
            int k = P.size() + 2;
            long long not_contained = 0;
            bool found = false;

            // --- NEW: Priority Check for Hardcoded Data ---
            if (static_table.count({P, t})) {
                not_contained = static_table[{P, t}];
                found = true;
            } 
            else if (memo_results.count({P, t})) {
                not_contained = memo_results[{P, t}];
                found = true;
            }

            // --- Logic Selection ---
            if (found) {
                // Already found in static_table or memo, skip to output
            }
            else if (P.empty()) {
                not_contained = 0;
            }
            else if (P.size() == 1 && (P[0] == 1 || P[0] == 3)) {
                not_contained = 1LL << (t - 2);
            }
            else if (P.size() == 1 && P[0] == 2) {
                not_contained = 0;
            }
            else if (is_star(P)) {
                not_contained = expression(P.size() + 1, t);
            }
            else if (P.size() == t-2){
                not_contained = total_trees - 1; // All trees except the one corresponding to P
            }
            else {
                // --- Precompute P structure (Threads will read this, so it's safe) ---
                auto edges_P = prufer_to_tree(P, k);
                vector<vector<int>> adj_P(k + 1);
                for (auto e : edges_P) {
                    adj_P[e.first].push_back(e.second);
                    adj_P[e.second].push_back(e.first);
                }
                vector<vector<int>> children_P(k + 1);
                vector<bool> visited_P(k + 1, false);
                queue<int> q_p;
                q_p.push(1);
                visited_P[1] = true;
                while (!q_p.empty()) {
                    int curr = q_p.front(); q_p.pop();
                    for (int nxt : adj_P[curr]) {
                        if (!visited_P[nxt]) {
                            visited_P[nxt] = true;
                            children_P[curr].push_back(nxt);
                            q_p.push(nxt);
                        }
                    }
                }
                int root_P_mask = 0;
                for (int child : children_P[1]) root_P_mask |= (1 << (child - 1));

                // --- HIGH-SPEED PARALLEL BRUTE FORCE ---
                long long global_tree_counter = 0;
                long long report_interval = total_trees / 100;
                if (report_interval < 1) report_interval = 1;

                #pragma omp parallel reduction(+:not_contained)
                {
                    vector<int> T_local(t - 2);
                    long long local_counter = 0;

                    #pragma omp for nowait
                    for (long long i = 0; i < total_trees; ++i) {
                        // 1. Reconstruct sequence i
                        long long temp_i = i;
                        for (int j = (t - 3); j >= 0; --j) {
                            T_local[j] = (temp_i % t) + 1;
                            temp_i /= t;
                        }

                        // 2. Minor Check 
                        if (!is_minor(T_local, children_P, root_P_mask, k)) {
                            not_contained++;
                        }

                        // 3. Optimized Progress Reporting
                        local_counter++;
                        if (local_counter >= 10000) { 
                            #pragma omp atomic
                            global_tree_counter += local_counter;
                            local_counter = 0;

                            if (omp_get_thread_num() == 0) {
                                double percent = (double)global_tree_counter / total_trees * 100.0;
                                cout << "\r> Progress: " << (int)percent << "% [" 
                                     << global_tree_counter << " / " << total_trees << "]" << flush;
                            }
                        }
                    }
                }
                cout << "\r> Progress: 100% [" << total_trees << " / " << total_trees << "]" << endl;
                
                memo_results[{P, t}] = not_contained;
                memo_results[{get_symmetric(P), t}] = not_contained;
            }

            // --- Final Clean Output ---
            cout << "N([";
            for (size_t z = 0; z < P.size(); ++z) cout << P[z] << (z + 1 == P.size() ? "" : ", ");
            cout << "], " << t << ") = " << not_contained << endl;
        }
    }
    return 0;
}