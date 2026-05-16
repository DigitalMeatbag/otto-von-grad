#ifndef MLP_H
#define MLP_H

#include "value.h"

typedef struct {
    int    nin;
    Value **w;   // [nin] weights
    Value  *b;   // bias
} Neuron;

typedef struct {
    int     nin;
    int     nout;
    Neuron *neurons;  // [nout]
} Layer;

typedef struct {
    int    nlayers;
    Layer *layers;
} MLP;

Neuron  neuron_create(Graph *g, int nin);
Value  *neuron_forward(Graph *g, Neuron *n, Value **x);

Layer   layer_create(Graph *g, int nin, int nout);
Value **layer_forward(Graph *g, Layer *l, Value **x);  // caller must free returned array

MLP     mlp_create(Graph *g, int nin, const int *sizes, int nlayers);
Value **mlp_forward(Graph *g, MLP *m, Value **x);      // caller must free returned array
void    mlp_free(MLP *m);

// Returns malloc'd array of all parameter Value*; caller must free the array (not the Values)
Value **mlp_parameters(MLP *m, int *out_count);

#endif
