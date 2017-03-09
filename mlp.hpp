#ifndef __MLP__
#define __MLP__

#include <iostream>
#include "Eigen/Core"

using namespace std;
class mlp
{
	private:
		int n_in_;
		int n_hid_;
		int n_out_;

		Eigen::VectorXd x_in_;
		Eigen::VectorXd x_hid_;
		Eigen::VectorXd x_out_;

		Eigen::MatrixXd w_in2hid_;
		Eigen::MatrixXd w_hid2out_;

		Eigen::MatrixXd delta_in2hid_;
		Eigen::MatrixXd delta_hid2out_;

		double active(double x)
		{
			return 1.0/(1.0+exp(-x));
		}

		double d_active(double fx)//活性化関数の返り値を渡し微分した値を返す
		{
			//double fx=active(x);dd
			return (1-fx)*fx;
		}

	public:
		mlp(int in,int hid,int out)
		{
			n_in_  = in;
			n_hid_ = hid;
			n_out_ = out;

			x_in_  = Eigen::VectorXd::Ones(n_in_+1);
			x_hid_ = Eigen::VectorXd::Ones(n_hid_+1);
			x_out_ = Eigen::VectorXd::Ones(n_out_);

			w_in2hid_  = Eigen::MatrixXd::Random(n_hid_,n_in_+1);
			w_hid2out_ = Eigen::MatrixXd::Random(n_out_,n_hid_+1);

		}

		Eigen::VectorXd forward(Eigen::VectorXd x)
		{
			for(int i=0;i<n_in_;i++)
				x_in_(i)=x(i);

			Eigen::VectorXd wx;

			wx=w_in2hid_*x_in_;
			for(int i=0;i<n_hid_;i++)
				x_hid_(i)=active(wx(i));

			wx=w_hid2out_*x_hid_;
			for(int i=0;i<n_out_;i++)
				x_out_(i)=active(wx(i));

			return x_out_; 
		}

		Eigen::VectorXd run(Eigen::VectorXd x,Eigen::VectorXd ty)
		{
			forward(x);

			Eigen::VectorXd dE_do=x_out_-ty;

			Eigen::VectorXd do_du(n_out_);
			for(int i=0;i<n_out_;i++)
				do_du(i)=d_active(x_out_(i));

			Eigen::VectorXd dE_du=dE_do.array()*do_du.array();

			delta_hid2out_=dE_du*x_hid_.transpose();

			dE_do=(dE_du.transpose()*w_hid2out_).transpose();

			dE_du.resize(n_hid_);
			for(int i=0;i<n_hid_;i++)
				dE_du(i)=dE_do(i)*d_active(x_hid_(i));

			delta_in2hid_=dE_du*x_in_.transpose();

			w_hid2out_-=delta_hid2out_;
			w_in2hid_-=delta_in2hid_;

			return x_out_; 
		}

};
#endif
