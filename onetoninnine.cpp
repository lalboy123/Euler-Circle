#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <queue>
#include <omp.h>

using namespace std;

// ========================================================================
// Math Utility Functions (Memoized for speed)
// ========================================================================

long long power(long long base, int exp) {
    long long res = 1;
    while (exp > 0) {
        if (exp % 2 == 1) res *= base;
        base *= base;
        exp /= 2;
    }
    return res;
}

long long fact(int n) {
    static vector<long long> f = {1, 1};
    while (f.size() <= n) f.push_back(f.back() * f.size());
    return f[n];
}

long long nCr(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fact(n) / (fact(k) * fact(n - k));
}

long long stirling2(int n, int k) {
    if (n == 0 && k == 0) return 1;
    if (n == 0 || k == 0) return 0;
    static vector<vector<long long>> S(15, vector<long long>(15, -1));
    if (n < 15 && k < 15 && S[n][k] != -1) return S[n][k];
    
    long long res = 0;
    for (int j = 0; j <= k; j++) {
        long long term = power(j, n) * nCr(k, j);
        if ((k - j) % 2 == 1) res -= term;
        else res += term;
    }
    res /= fact(k);
    if (n < 15 && k < 15) S[n][k] = res;
    return res;
}

// ========================================================================
// User Provided Logic & Expressions
// ========================================================================

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

// ========================================================================
// Graph Construction & Structural Tools
// ========================================================================

vector<vector<int>> prufer_to_tree(const vector<int>& prufer, int nodes) {
    vector<vector<int>> adj(nodes + 1);
    vector<int> degree(nodes + 1, 1);
    for (int p : prufer) degree[p]++;
    
    int ptr = 1;
    while (ptr <= nodes && degree[ptr] != 1) ptr++;
    int leaf = ptr;
    
    for (int p : prufer) {
        adj[leaf].push_back(p);
        adj[p].push_back(leaf);
        degree[leaf]--;
        degree[p]--;
        if (p < ptr && degree[p] == 1) {
            leaf = p;
        } else {
            ptr++;
            while (ptr <= nodes && degree[ptr] != 1) ptr++;
            leaf = ptr;
        }
    }
    adj[leaf].push_back(nodes);
    adj[nodes].push_back(leaf);
    return adj;
}

int get_centroid(const vector<vector<int>>& adj) {
    int n = adj.size() - 1;
    vector<int> sz(n + 1, 0);
    int centroid = -1;
    
    auto dfs = [&](auto& self, int u, int p) -> void {
        sz[u] = 1;
        bool is_centroid = true;
        for (int v : adj[u]) {
            if (v != p) {
                self(self, v, u);
                sz[u] += sz[v];
                if (sz[v] > n / 2) is_centroid = false;
            }
        }
        if (n - sz[u] > n / 2) is_centroid = false;
        if (is_centroid) centroid = u;
    };
    dfs(dfs, 1, 0);
    return centroid != -1 ? centroid : 1;
}

// ========================================================================
// Bipartite Matching for Branch Assignments
// ========================================================================

bool bpm(int u, const vector<vector<int>>& bpGraph, vector<int>& match, vector<bool>& seen) {
    for (int v : bpGraph[u]) {
        if (!seen[v]) {
            seen[v] = true;
            if (match[v] < 0 || bpm(match[v], bpGraph, match, seen)) {
                match[v] = u;
                return true;
            }
        }
    }
    return false;
}

// ========================================================================
// Main DP Minor Check Algorithm
// ========================================================================

bool is_minor(const vector<int>& P_seq, const vector<int>& T_seq, int p_nodes, int t_nodes) {
    auto P_adj = prufer_to_tree(P_seq, p_nodes);
    auto T_adj = prufer_to_tree(T_seq, t_nodes);
    
    // 1. Centroid Rooting for P
    int root_P = get_centroid(P_adj);
    
    // Fast Leaf Extraction
    vector<bool> is_leaf_P(p_nodes + 1, true);
    for(int x : P_seq) is_leaf_P[x] = false;

    // Try rooting T at every possible vertex
    for (int root_T = 1; root_T <= t_nodes; root_T++) {
        
        vector<int> min_label(t_nodes + 1), max_label(t_nodes + 1);
        vector<vector<int>> children_T(t_nodes + 1);
        vector<int> post_order_T;
        
        // 2. Build T hierarchies and Precompute Label Bounds
        auto dfs_T = [&](auto& self, int u, int p) -> void {
            min_label[u] = u;
            max_label[u] = u;
            for (int v : T_adj[u]) {
                if (v != p) {
                    children_T[u].push_back(v);
                    self(self, v, u);
                    min_label[u] = min(min_label[u], min_label[v]);
                    max_label[u] = max(max_label[u], max_label[v]);
                }
            }
            post_order_T.push_back(u);
        };
        dfs_T(dfs_T, root_T, 0);
        
        vector<vector<int>> children_P(p_nodes + 1);
        vector<int> post_order_P;
        auto dfs_P = [&](auto& self, int u, int p) -> void {
            for (int v : P_adj[u]) {
                if (v != p) {
                    children_P[u].push_back(v);
                    self(self, v, u);
                }
            }
            post_order_P.push_back(u);
        };
        dfs_P(dfs_P, root_P, 0);

        // DP Tables: Valid[t][p] and DescendantValid[t][p]
        vector<vector<bool>> Valid(t_nodes + 1, vector<bool>(p_nodes + 1, false));
        vector<vector<bool>> DescValid(t_nodes + 1, vector<bool>(p_nodes + 1, false));
        
        for (int t : post_order_T) {
            for (int p : post_order_P) {
                // Base Case: p is a leaf in P
                if (is_leaf_P[p]) {
                    Valid[t][p] = true;
                } else {
                    // 3. Bipartite Matching Setup
                    int k_children = children_P[p].size();
                    int t_branches = children_T[t].size();
                    
                    if (t_branches < k_children) {
                        Valid[t][p] = false;
                    } else {
                        vector<vector<int>> bpGraph(k_children);
                        for (int i = 0; i < k_children; i++) {
                            int cp = children_P[p][i];
                            for (int j = 0; j < t_branches; j++) {
                                int ct = children_T[t][j];
                                
                                // O(1) Label Inequality Pruning
                                // If child label implies a constraint, ensure T's branch can satisfy it
                                if (cp > p && max_label[ct] < cp) continue;
                                if (cp < p && min_label[ct] > cp) continue;
                                
                                if (DescValid[ct][cp]) {
                                    bpGraph[i].push_back(j);
                                }
                            }
                        }
                        
                        // Execute Bipartite Matching
                        vector<int> match(t_branches, -1);
                        int matches = 0;
                        for (int i = 0; i < k_children; i++) {
                            vector<bool> seen(t_branches, false);
                            if (bpm(i, bpGraph, match, seen)) matches++;
                        }
                        Valid[t][p] = (matches == k_children);
                    }
                }
                
                DescValid[t][p] = Valid[t][p];
                for (int ct : children_T[t]) {
                    if (DescValid[ct][p]) {
                        DescValid[t][p] = true;
                        break;
                    }
                }
            }
        }
        
        if (Valid[root_T][root_P]) return true;
    }
    return false;
}

// ========================================================================
// Main Routine
// ========================================================================

int main() {
    // Environment setup for testing
    int t = 10; // Vertices in T
    long long total_trees = power(t, t - 2);
    
    vector<int> P = {1, 3}; // Example Prufer sequence for P
    long long not_contained = -1;
    bool found = false;

    // --- INSERT BIG STATIC TABLE HERE ---
    // Example format:
    // if (t == 10 && P == vector<int>{1, 2, 3}) { not_contained = 123456; found = true; }
    
    // ------------------------------------

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
    else if (P.size() == t - 2){
        not_contained = total_trees - 1; 
    }
    else {
        // Parallelized Brute Force with Mathematical Pruning
        int p_nodes = P.size() + 2;
        long long local_not_contained = 0;
        
        // Example over a subset of Prufer sequences for T. 
        // In reality, you'd iterate over all t^(t-2) sequences here.
        #pragma omp parallel for reduction(+:local_not_contained) schedule(dynamic)
        for (long long i = 0; i < total_trees; i++) {
            // Reconstruct T sequence from index 'i'
            vector<int> T_seq(t - 2);
            long long temp = i;
            for(int j = 0; j < t - 2; j++) {
                T_seq[j] = (temp % t) + 1;
                temp /= t;
            }
            
            if (!is_minor(P, T_seq, p_nodes, t)) {
                local_not_contained++;
            }
        }
        not_contained = local_not_contained;
    }

    cout << "N([";
    for (size_t z = 0; z < P.size(); ++z) cout << P[z] << (z + 1 == P.size() ? "" : ", ");
    cout << "], " << t << ") = " << not_contained << endl;
}