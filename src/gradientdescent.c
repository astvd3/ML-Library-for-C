#include "gradientdescent.h"

/* Declaration of the global weights */
typedef struct
{
    float theta0,theta1;
}theta_t;

static theta_t linreg_globalTheta;

/* @brief : Generates hypothesis from the existing weights
 *
 * @param [in] data Output of the estimated input data
   @return hypothesis output of the data
 */
static float linreg_hypothesis(float data)
{
    return (linreg_globalTheta.theta0 + data*linreg_globalTheta.theta1);
}

/*
 * @brief : Calculates cost from gradient descent cost function
 *
 * @param [in] x Data for input vector
 * @param [in] y Data for output vector
 * @param [in] num_samples Number of samples in the data
 * @param [out] sumerr0 Sum_error 0 for weight theta0
 * @param [out] sumerr1 Sum error 1 for weight theta1
 * @return sumsqerr square sum of error
 */
static float linreg_costfunction(float *x, float *y,uint16_t num_samples,float *sumerr0,float *sumerr1){
	int i;
	float sumsqerr = 0;
    float err[num_samples];
    *sumerr0=0;
    *sumerr1=0;

    for(i=0;i<(int)num_samples;i++)
    {
        err[i]=linreg_hypothesis(x[i]) - y[i];
        sumsqerr += (err[i] * err[i]);
        (*sumerr0) += err[i];
        (*sumerr1) += (err[i] * x[i]);
    }
	sumsqerr=sumsqerr/(2.0f*(float)num_samples);
	return sumsqerr;
}

/*
 * @brief : Uses Gradient descent algorithm to readjust the weights
 *
 * @param [in] alpha learning rate for weight adjustment
 * @param [in] num_samples Number of samples in the data
 * @param [in] sumerr0 Sum_error 0 for weight theta0
 * @param [in] sumerr1 Sum error 1 for weight theta1
 * @return Void
 */
static void linreg_gradientDescent(float alpha,uint16_t num_samples,float sumerr0,float sumerr1)
{
    linreg_globalTheta.theta0=linreg_globalTheta.theta0 - ((alpha*sumerr0)/(float)num_samples);
    linreg_globalTheta.theta1=linreg_globalTheta.theta1 - ((alpha*sumerr1)/(float)num_samples);
}

/*
 * @brief : Starts Linear Regulation on the provided data
 *
 * Size of X and Size of Y should be the same
 * @param [in] x Data for input vector
 * @param [in] y Data for output vector
 * @param [in] num_iters Number of iterations to calculate weights
 * @param [in] alpha learmimg rate for gradient descent
 * @param [in] num_samples Number of samples to train on
 * @return Void
 */
void linreg_regressionRun(float* x,float* y,uint16_t num_iters,float alpha,uint16_t num_samples)
{
    float cost = 0;
    float sumerr0 = 0;
    float sumerr1 = 0;

    for(int i = 0; i < num_iters;i++)
    {
        linreg_gradientDescent(alpha,num_samples,sumerr0,sumerr1);
        cost=linreg_costfunction(x,y,num_samples,&sumerr0,&sumerr1);
    }
}

/*
 * @brief : Retrieves Weights on Linear Regulators at any point of the processing
 *
 * @param [out] theta0 value of theta0 in float
 * @param [out] theta1 value of theta1 in float
 * @return Void
 */
void linreg_getWeights(float* theta0,float* theta1)
{
    *theta0 = linreg_globalTheta.theta0;
    *theta1 = linreg_globalTheta.theta1;
}

/*
 * @brief : Initializes global weights
 *
 * If this function is not used, weights are initialized at zero
 * @param [in] theta0 value of theta0 in float
 * @param [in] theta1 value of theta1 in float
 * @return Void
 */
void linreg_initWeights(float theta0,float theta1)
{
    linreg_globalTheta.theta0 = theta0;
    linreg_globalTheta.theta1 = theta1;
}
