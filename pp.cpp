#include<iostream>
using namespace std;
class con{
    int add,sub;
    public:
    con(int a,int b){
        add=a;
        sub=b;
    }
    void show(){
        cout<<add+sub<<endl;
    }
    con(con &c){
        add=c.add;
        sub=c.sub;
    }
};
int main(){
    con c1(45,8);
    con c2=c1;
    c1.show();
    c2.show();
}