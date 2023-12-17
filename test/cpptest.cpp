#include<iostream>
#include<queue>
#include<vector>
#include<stack>
using namespace std;

struct CompareIndex {
    const std::vector<int> &array;
    CompareIndex(const std::vector<int> &arr) : array(arr) {}
    //自定义比较函数
    bool operator()(int i, int j) const {return array[i] > array[j];}
};

int main() {
    vector<int> nums = {1, 2, 3, 4, 5, 6};
    stack<int> s;
    priority_queue<int,vector<int>, CompareIndex> q((CompareIndex(nums)));
    cout << typeid(CompareIndex(nums)).name() << endl;
    cout << typeid((CompareIndex(nums))).name() << endl;

    int n = nums.size();
    vector<int> res(n,-1);
    for (int i = 0; i < n; i++) 
    {
        while (!q.empty() && nums[i] > nums[q.top()]) 
        {
            res[q.top()] = nums[i];
            q.pop();
        }
        while (!s.empty() && nums[i] > nums[s .top()]) {
            q.push(s.top());
            s.pop();
        }
        s.push(i);
    }
}