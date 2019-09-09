#include "neuralnetwork.h"

#define MAX_NUM_LAYERS  (5)
#define MAX_IP_LAYER    (1000)

static nn_t global_net;
static float hidden_layerConfig[MAX_NUM_LAYERS][2];
static float global_weightsArray[MAX_NUM_LAYERS][MAX_IP_LAYER]; // TODO : Handle this dynamically to avoid wastage of memory

void nn_init(uint16_t input_layer_size)
{
    global_net.input_layer_size = input_layer_size;
    global_weightsArray[0][0]= 0;
}

void nn_addHiddenLayer(uint16_t hidden_layer_neurons,actfunc_t activation_function)
{
    hidden_layerConfig[global_net.number_of_hidden_layers][0] = hidden_layer_neurons;
    hidden_layerConfig[global_net.number_of_hidden_layers][1] = activation_function;
    global_net.number_of_hidden_layers++;
}

void nn_addOutputLayer(uint16_t output_layer_size,actfunc_t activation_function)
{
    hidden_layerConfig[global_net.number_of_hidden_layers][0] = output_layer_size;
    hidden_layerConfig[global_net.number_of_hidden_layers][1] = activation_function;
    global_net.number_of_hidden_layers++;
    //set the nn;
    global_net.output_layer_size = output_layer_size;
    printf("******Neural Network Configuration*****\n");
    printf("Input dim %d number of hidden_layers %d output size %d\n",global_net.input_layer_size,
                    global_net.number_of_hidden_layers,global_net.output_layer_size);
    printf("******Neural Network Configuration*****\n");
}
