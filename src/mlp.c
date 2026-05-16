#include "mlp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;  // uniform [-1, 1]
}

// ── Neuron ───────────────────────────────────────────────────────────────────

Neuron neuron_create(Graph *g, int nin) {
    Neuron n;
    n.nin = nin;
    n.w   = malloc(nin * sizeof(Value *));
    if (!n.w) { fprintf(stderr, "neuron_create: out of memory\n"); exit(1); }
    for (int i = 0; i < nin; i++) n.w[i] = val_new(g, randf());
    n.b = val_new(g, 0.0f);
    return n;
}

// out = tanh(w[0]*x[0] + w[1]*x[1] + ... + b)
Value *neuron_forward(Graph *g, Neuron *n, Value **x) {
    Value *sum = n->b;
    for (int i = 0; i < n->nin; i++) {
        Value *wx = val_mul(g, n->w[i], x[i]);
        sum = val_add(g, sum, wx);
    }
    return val_tanh(g, sum);
}

// ── Layer ────────────────────────────────────────────────────────────────────

Layer layer_create(Graph *g, int nin, int nout) {
    Layer l;
    l.nin     = nin;
    l.nout    = nout;
    l.neurons = malloc(nout * sizeof(Neuron));
    if (!l.neurons) { fprintf(stderr, "layer_create: out of memory\n"); exit(1); }
    for (int i = 0; i < nout; i++) l.neurons[i] = neuron_create(g, nin);
    return l;
}

Value **layer_forward(Graph *g, Layer *l, Value **x) {
    Value **out = malloc(l->nout * sizeof(Value *));
    if (!out) { fprintf(stderr, "layer_forward: out of memory\n"); exit(1); }
    for (int i = 0; i < l->nout; i++) out[i] = neuron_forward(g, &l->neurons[i], x);
    return out;
}

// ── MLP ──────────────────────────────────────────────────────────────────────

// sizes: output size of each layer, e.g. {4, 1} for 2→4→1
MLP mlp_create(Graph *g, int nin, const int *sizes, int nlayers) {
    MLP m;
    m.nlayers = nlayers;
    m.layers  = malloc(nlayers * sizeof(Layer));
    if (!m.layers) { fprintf(stderr, "mlp_create: out of memory\n"); exit(1); }
    int in = nin;
    for (int i = 0; i < nlayers; i++) {
        m.layers[i] = layer_create(g, in, sizes[i]);
        in = sizes[i];
    }
    return m;
}

// Chains layers; frees intermediate output arrays; caller must free the final result
Value **mlp_forward(Graph *g, MLP *m, Value **x) {
    Value **current = x;
    for (int i = 0; i < m->nlayers; i++) {
        Value **next = layer_forward(g, &m->layers[i], current);
        if (i > 0) free(current);  // free intermediate (never free the caller's input)
        current = next;
    }
    return current;
}

void mlp_free(MLP *m) {
    for (int l = 0; l < m->nlayers; l++) {
        for (int n = 0; n < m->layers[l].nout; n++) free(m->layers[l].neurons[n].w);
        free(m->layers[l].neurons);
    }
    free(m->layers);
}

Value **mlp_parameters(MLP *m, int *out_count) {
    int count = 0;
    for (int l = 0; l < m->nlayers; l++)
        for (int n = 0; n < m->layers[l].nout; n++)
            count += m->layers[l].neurons[n].nin + 1;  // weights + bias

    Value **params = malloc(count * sizeof(Value *));
    if (!params) { fprintf(stderr, "mlp_parameters: out of memory\n"); exit(1); }

    int idx = 0;
    for (int l = 0; l < m->nlayers; l++)
        for (int n = 0; n < m->layers[l].nout; n++) {
            Neuron *neuron = &m->layers[l].neurons[n];
            for (int i = 0; i < neuron->nin; i++) params[idx++] = neuron->w[i];
            params[idx++] = neuron->b;
        }

    *out_count = count;
    return params;
}
