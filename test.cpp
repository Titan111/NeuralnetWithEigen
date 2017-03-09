#include <iostream>
#include <math.h>
#include "Eigen/Core"
#include "mlp.hpp"

using namespace std;
using namespace Eigen;

double sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

int main()
{
	VectorXd x[4];
	VectorXd t[4];
	for(int i=0;i<4;i++)
	{
		x[i].resize(2);
		t[i].resize(1);
	}
	x[0]<<0,0;
	t[0]<<0;
	x[1]<<1,0;
	t[1]<<1;
	x[2]<<0,1;
	t[2]<<1;
	x[3]<<1,1;
	t[3]<<0;
	cout<<"mlp class test"<<endl;

	mlp nn(2,3,1);
	for(int epoch=0;epoch<10000;epoch++)
	{
		for(int i=0;i<4;i++)
		{
			Eigen::VectorXd ans = nn.run(x[i],t[i]);

			cout<<"i:"<<i<<"\t"<<ans<<endl;
		}
	}
	return 0;
}

