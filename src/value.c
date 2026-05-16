#include "value.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Arena ────────────────────────────────────────────────────────────────────

Graph *graph_create(void) {
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) { fprintf(stderr, "graph_create: out of memory\n"); exit(1); }
    return g;
}

void graph_free(Graph *g) {
    free(g);
}

void graph_zero_grad(Graph *g) {
    for (int i = 0; i < g->count; i++) g->nodes[i].grad = 0.0f;
}

void graph_reset_to(Graph *g, int checkpoint) {
    g->count = checkpoint;
    for (int i = 0; i < checkpoint; i++) g->nodes[i].grad = 0.0f;
}

// Allocate the next node from the arena
static Value *alloc_node(Graph *g) {
    if (g->count >= MAX_NODES) {
        fprintf(stderr, "graph: MAX_NODES (%d) exceeded\n", MAX_NODES);
        exit(1);
    }
    Value *v = &g->nodes[g->count++];
    memset(v, 0, sizeof(Value));
    return v;
}

// ── Leaf ─────────────────────────────────────────────────────────────────────

Value *val_new(Graph *g, float data) {
    Value *v = alloc_node(g);
    v->data = data;
    strncpy(v->op, "leaf", sizeof(v->op));
    return v;
}

// ── Backward functions ───────────────────────────────────────────────────────

static void backward_add(Value *self) {
    // d/da (a+b) = 1, d/db (a+b) = 1
    self->children[0]->grad += self->grad;
    self->children[1]->grad += self->grad;
}

static void backward_mul(Value *self) {
    // d/da (a*b) = b, d/db (a*b) = a
    self->children[0]->grad += self->children[1]->data * self->grad;
    self->children[1]->grad += self->children[0]->data * self->grad;
}

static void backward_sub(Value *self) {
    // d/da (a-b) = 1, d/db (a-b) = -1
    self->children[0]->grad += self->grad;
    self->children[1]->grad -= self->grad;
}

static void backward_pow(Value *self) {
    // d/dx x^p = p * x^(p-1)
    float p = self->aux;
    self->children[0]->grad += p * powf(self->children[0]->data, p - 1.0f) * self->grad;
}

static void backward_tanh(Value *self) {
    // d/dx tanh(x) = 1 - tanh(x)^2
    float t = self->data;
    self->children[0]->grad += (1.0f - t * t) * self->grad;
}

// ── Operations ───────────────────────────────────────────────────────────────

Value *val_add(Graph *g, Value *a, Value *b) {
    Value *out = alloc_node(g);
    out->data        = a->data + b->data;
    out->children[0] = a;
    out->children[1] = b;
    out->n_children  = 2;
    out->backward    = backward_add;
    strncpy(out->op, "+", sizeof(out->op));
    return out;
}

Value *val_mul(Graph *g, Value *a, Value *b) {
    Value *out = alloc_node(g);
    out->data        = a->data * b->data;
    out->children[0] = a;
    out->children[1] = b;
    out->n_children  = 2;
    out->backward    = backward_mul;
    strncpy(out->op, "*", sizeof(out->op));
    return out;
}

Value *val_sub(Graph *g, Value *a, Value *b) {
    Value *out = alloc_node(g);
    out->data        = a->data - b->data;
    out->children[0] = a;
    out->children[1] = b;
    out->n_children  = 2;
    out->backward    = backward_sub;
    strncpy(out->op, "-", sizeof(out->op));
    return out;
}

Value *val_pow(Graph *g, Value *a, float p) {
    Value *out = alloc_node(g);
    out->data        = powf(a->data, p);
    out->aux         = p;
    out->children[0] = a;
    out->n_children  = 1;
    out->backward    = backward_pow;
    strncpy(out->op, "^", sizeof(out->op));
    return out;
}

Value *val_tanh(Graph *g, Value *a) {
    Value *out = alloc_node(g);
    out->data        = tanhf(a->data);
    out->children[0] = a;
    out->n_children  = 1;
    out->backward    = backward_tanh;
    strncpy(out->op, "tanh", sizeof(out->op));
    return out;
}

// ── Backward pass ────────────────────────────────────────────────────────────

// Recursive topo sort: post-order DFS, visited tracked by pointer scan
static void topo_sort(Value *v,
                      Value **topo,  int *topo_n,
                      Value **seen,  int *seen_n) {
    for (int i = 0; i < *seen_n; i++) if (seen[i] == v) return;
    seen[(*seen_n)++] = v;
    for (int i = 0; i < v->n_children; i++)
        topo_sort(v->children[i], topo, topo_n, seen, seen_n);
    topo[(*topo_n)++] = v;
}

void val_backward(Value *root) {
    Value *topo[MAX_NODES];
    Value *seen[MAX_NODES];
    int topo_n = 0, seen_n = 0;

    topo_sort(root, topo, &topo_n, seen, &seen_n);

    root->grad = 1.0f;
    for (int i = topo_n - 1; i >= 0; i--)
        if (topo[i]->backward) topo[i]->backward(topo[i]);
}

// ── Debug ────────────────────────────────────────────────────────────────────

void val_print(const Value *v) {
    printf("Value(op=%-4s  data=%8.4f  grad=%8.4f)\n", v->op, v->data, v->grad);
}
