#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <tuple>
#include <omp.h>
#include <mutex>

using namespace std;

std::mutex mtx; 

// --- Global Constants ---
const int MAX_N = 15;
const int MAX_K = 10; 
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

// Pareto-Dominance check: Returns true if 'a' is strictly better than 'b'
bool dominates(const Summary& a, const Summary& b, int k) {
    bool strictly_better = false;
    for (int i = 0; i < k; ++i) {
        if (a.min_v[i] == 0 && b.min_v[i] == 0) continue;
        // If a requires a larger minimum or produces a smaller maximum, it doesn't dominate
        if (a.min_v[i] > b.min_v[i] || a.max_v[i] < b.max_v[i]) return false;
        // If a has a strictly smaller min or strictly larger max, it is better
        if (a.min_v[i] < b.min_v[i] || a.max_v[i] > b.max_v[i]) strictly_better = true;
    }
    return strictly_better;
}

// --- Thread-Local Storage ---
thread_local bool dp_visited[MAX_N][MAX_K][MAX_MASK];
thread_local vector<tuple<int, int, int>> dp_touched;
thread_local vector<Summary> dp_memo[MAX_N][MAX_K][MAX_MASK];
static thread_local vector<int> leaves_T; // Used for structural pruning

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
    for (int i = 1; i <= r; i++) res = res * (n - i + 1) / i;
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
    for (int i = 0; i < n; ++i) sym[i] = n + 3 - seq[n - 1 - i];
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

void decode_prufer(long long idx, int t, vector<int>& res) {
    for (int i = 0; i < t - 2; ++i) {
        res[i] = (idx % t) + 1;
        idx /= t;
    }
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
            // MATH SPEEDUP #1: TOPOLOGICAL LEAF PRUNING
            // If the host subtree doesn't have enough leaves for the requested branches, skip.
            if (leaves_T[v] < __builtin_popcount(S)) {
                if (S == 0) break;
                S = (S - 1) & R_mask;
                continue;
            }

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

    // MATH SPEEDUP #2: PARETO-DOMINANCE FILTERING
    // Strip away mathematically inferior sub-summaries
    vector<Summary> pareto_filtered;
    for (size_t i = 0; i < all_mappings.size(); ++i) {
        bool is_dominated = false;
        for (size_t j = 0; j < all_mappings.size(); ++j) {
            if (i == j) continue;
            if (dominates(all_mappings[j], all_mappings[i], k)) {
                is_dominated = true;
                break;
            }
        }
        if (!is_dominated) pareto_filtered.push_back(all_mappings[i]);
    }

    res = pareto_filtered;
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
        leaves_T.assign(n + 1, 0); // Initialize leaf tracking
        while(!qt.empty()) qt.pop();
        
        qt.push(r_T);
        visited_T[r_T] = true;
        vector<int> rev_order; // Used to compute leaves bottom-up

        while (!qt.empty()) {
            int curr = qt.front(); qt.pop();
            rev_order.push_back(curr);
            for (int nxt : adj_T[curr]) {
                if (!visited_T[nxt]) {
                    visited_T[nxt] = true;
                    children_T[curr].push_back(nxt);
                    qt.push(nxt);
                }
            }
        }

        // Compute Leaf Counts bottom-up
        for (int i = (int)rev_order.size() - 1; i >= 0; --i) {
            int u = rev_order[i];
            if (children_T[u].empty()) leaves_T[u] = 1;
            else {
                for (int v : children_T[u]) leaves_T[u] += leaves_T[v];
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
    // Data omitted for space but preserved logic
}

int main() {
    // --- Fast I/O ---
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    load_table();

    int t = 12; // Adjust target 't' here
    // Cayley's Formula: t^(t-2)
    long long total_trees = (t == 2) ? 1 : 1;
    if (t > 2) {
        for(int i = 0; i < t - 2; ++i) total_trees *= t;
    }

    map<pair<vector<int>, int>, long long> memo_results;

    cout << "Target t = " << t << " | Total Trees to check: " << total_trees << endl;
    cout << "--------------------------------------------------------" << endl;

    for (int len = 0; len <= t-2; ++len) { 
        auto P_seqs = all_prufer_sequences(len + 2); 
        for (const auto& P : P_seqs) {
            int k = P.size() + 2;
            long long not_contained = 0;
            bool found = false;

            if (static_table.count({P, t})) {
                not_contained = static_table[{P, t}];
                found = true;
            } 
            else if (memo_results.count({P, t})) {
                not_contained = memo_results[{P, t}];
                found = true;
            }

            if (found) {
                // Already found, bypass heavy logic
            }
            else if (P.empty()) not_contained = 0;
            else if (P.size() == 1 && (P[0] == 1 || P[0] == 3)) not_contained = 1LL << (t - 2);
            else if (P.size() == 1 && P[0] == 2) not_contained = 0;
            else if (is_star(P)) not_contained = expression(P.size() + 1, t);
            else if (P.size() == t-2) not_contained = total_trees - 1; 
            else {
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

                long long global_tree_counter = 0;
                
                #pragma omp parallel reduction(+:not_contained)
                {
                    vector<int> T_local(t - 2);
                    long long local_counter = 0;

                    #pragma omp for nowait
                    for (long long i = 0; i < total_trees; ++i) {
                        decode_prufer(i, t, T_local);

                        if (!is_minor(T_local, children_P, root_P_mask, k)) {
                            not_contained++;
                        }

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

            cout << "N([";
            for (size_t z = 0; z < P.size(); ++z) cout << P[z] << (z + 1 == P.size() ? "" : ", ");
            cout << "], " << t << ") = " << not_contained << endl;
        }
    }
    return 0;
}