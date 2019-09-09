#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdint.h>

typedef enum
{
    ACT_RELU,
    ACT_LINEAR,
    ACT_SIGMOID,
    ACT_TANH
}actfunc_t;

typedef struct
{
    uint16_t input_layer_size;
    uint16_t number_of_hidden_layers;
    uint16_t output_layer_size;
    float **weight_matrix;
    actfunc_t **actfunc_set;
}nn_t;

void nn_init(uint16_t input_layer_size);
void nn_addHiddenLayer(uint16_t hidden_layer_neurons,actfunc_t activation_function);
void nn_addOutputLayer(uint16_t output_layer_size,actfunc_t activation_function);

#endif //NEURALNETWORK_H
