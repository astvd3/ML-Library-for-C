#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include <stdint.h>
#include <stdio.h>

void linreg_initWeights(float theta0,float theta1);
void linreg_getWeights(float* theta0,float* theta1);
void linreg_regressionRun(float* x,float* y,uint16_t num_iters,float alpha,uint16_t num_samples);

#endif //GRADIENTDESCENT_H
