#ifndef VALUE_H
#define VALUE_H

#define MAX_NODES    16384
#define MAX_CHILDREN 2

typedef struct Value Value;
typedef void (*BackwardFn)(Value *self);

struct Value {
    float      data;
    float      grad;
    float      aux;         // extra scalar for ops that need a parameter (e.g. pow exponent)
    Value     *children[MAX_CHILDREN];
    int        n_children;
    BackwardFn backward;
    char       op[8];
};

typedef struct {
    Value nodes[MAX_NODES];
    int   count;
} Graph;

Graph *graph_create(void);
void   graph_free(Graph *g);
void   graph_zero_grad(Graph *g);
void   graph_reset_to(Graph *g, int checkpoint);  // rewind arena to checkpoint, zero grads up to there

Value *val_new(Graph *g, float data);
Value *val_add(Graph *g, Value *a, Value *b);
Value *val_mul(Graph *g, Value *a, Value *b);
Value *val_sub(Graph *g, Value *a, Value *b);
Value *val_pow(Graph *g, Value *a, float p);
Value *val_tanh(Graph *g, Value *a);

void val_backward(Value *root);
void val_print(const Value *v);

#endif
