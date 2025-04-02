#include <vector>
using namespace std;

vector<int> elemwise_add(
    vector<int> tensor_x,
    vector<int> tensor_y
) {
    vector<int> out;
    int m = tensor_x.size();
    for (int i = 0; i < m; i++) {
        out.push_back(tensor_x[i] + tensor_y[i]);
    }
    return out;
}
