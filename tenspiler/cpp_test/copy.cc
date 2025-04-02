#include <vector>
using namespace std;

vector<int> copy(vector<int> tensor_x) {
    vector<int> out;
    int m = tensor_x.size();

    for (int i = 0; i < m; i++) {
        out.push_back(tensor_x[i]);
    }
    return out;
}
