#include <vector>
using namespace std;


vector<vector<int>> matmul(vector<vector<int>> input, vector<vector<int>> weight) {
    vector<vector<int>> output;
    int n = weight[1].size();
    int m = input.size();
    int k = input[1].size();
    for (int row = 0; row < m; row++) {
        vector<int> sum;
        for (int col = 0; col < n; col++) {
            int curr = 0;
            for (int j = 0; j < k; j++) {
                curr += input[row][j] * weight[j][col];
            }
            sum.push_back(curr);
        }
        output.push_back(sum);
    }
    return output;
}
