#include "mathmatical_functions.c"


/*
ニューラルネットワークの層および最適化関数を実装した.
*/


// ------ layers ------

// ReLU layer

typedef struct{
    int len;          // 入出力ノード数
    double *mask; // 傾き
}ReluLayer;

void relu_init(ReluLayer *layer, int len){
    // ReluLayerを初期化する.
    layer->len = len;
    layer->mask = malloc(len * sizeof(double));
}

void relu_forward(ReluLayer *layer, const double x[], double out[]){
    // 入力xを受け取って, outに結果を出力する.
    // x -- ReluLayer --> out
    for (int i = 0; i < layer->len; i++){
        if (x[i] <= 0.0){
            out[i] = 0.0;
            layer->mask[i] = 0.0;
        }else{
            out[i] = x[i];
            layer->mask[i] = 1.0;
        }
    }
}

void relu_backward(const ReluLayer *layer, const double dout[], double dx[]){
    // 勾配doutを受け取って, dxに結果を出力する.
    // dx <-- ReluLayer -- dout
    for (int i = 0; i < layer->len; i++)
        dx[i] = layer->mask[i] * dout[i];
}


// sigmoid layer

typedef struct{
    int len;     // 入出力ノード数
    double *out; // 出力結果
}SigmoidLayer;

void sigmoid_init(SigmoidLayer *layer, int len){
    // SigmoidLayerを初期化する.
    layer->len = len;
    layer->out = malloc(len * sizeof(double));
}

void sigmoid_forward(SigmoidLayer *layer, const double x[], double out[]){
    // 入力xを受け取って, outに結果を出力する.
    // x -- SigmoidLayer --> out
    for (int i = 0; i < layer->len; i++){
        out[i] = 1.0 / (1.0 + pow(M_E, -x[i]));
        layer->out[i] = out[i];
    }
}

void sigmoid_backward(const SigmoidLayer *layer, const double dout[], double dx[]){
    // 勾配doutを受け取って, dxに結果を出力する.
    // dx <-- SigmoidLayer -- dout
    for (int i = 0; i < layer->len; i++)
        dx[i] = dout[i];// * (1.0 - layer->out[i]) * layer->out[i];
}


// affine layer

typedef struct{
    int n;       // 入力ノード数
    int m;       // 出力ノード数
    double **w;  // weights(m x n)
    double *b;   // bias(m)
    double *x;   // 入力
    double **dw;
    double *db;
}AffineLayer;

void affine_init(AffineLayer *layer, int n, int m){
    // AffineLayerを初期化する.
    // 入力ノード数はn, 出力ノード数はmである.
    layer->n = n;
    layer->m = m;
    layer->w = malloc(m * sizeof(double*));
    matrix_init(layer->w, m, n);
    layer->b = malloc(m * sizeof(double));
    layer->x = malloc(n * sizeof(double));
    layer->dw = malloc(m * sizeof(double*));
    matrix_init(layer->dw, m, n);
    layer->db = malloc(m * sizeof(double));
}

void affine_init_with_xavier(AffineLayer *layer, int n, int m){
    // Xavierの初期値でAffineLayerを初期化する.
    affine_init(layer, n, m);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            layer->w[i][j] = xavier(n);
    }
}

void affine_init_with_he(AffineLayer *layer, int n, int m){
    // Heの初期値でAffineLayerを初期化する.
    affine_init(layer, n, m);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            layer->w[i][j] = he(n);
    }
}

void affine_forward(AffineLayer *layer, const double x[], double out[]){
    // 入力xを受け取って, outに結果を出力する.
    // x -- AffineLayer --> out
    for (int i = 0; i < layer->n; i++)
        layer->x[i] = x[i];
    mat_mul_vec(layer->w, x, out, layer->m, layer->n);
}

void affine_backward(AffineLayer *layer, const double dout[], double dx[]){
    // 勾配doutを受け取って, dxに結果を出力する.
    // dx <-- AffineLayer -- dout
    // dxを計算する.
    double **wt = transpose(layer->w, layer->m, layer->n);
    mat_mul_vec(wt, dout, dx, layer->n, layer->m);
    delete_matrix(wt, layer->n);
    // dwを計算する.
    vec_mul_vecT(dout, layer->x, layer->dw, layer->m, layer->n);
    // dbを計算する.
    for (int i = 0; i < layer->m; i++)
        layer->db[i] = dout[i];
}


// ------ optimizers ------

typedef struct{
    // 最適化関数で用いられる変数の定義
    double **wv;
    double *bv;
    double **ws;
    double *bs;
}Velocities;

void velocities_init(Velocities *v, const AffineLayer *layer){
    // Velocitiesを初期化する.
    v->wv = malloc(layer->m * sizeof(double*));
    for (int i = 0; i < layer->m; i++)
        v->wv[i] = calloc(layer->n, sizeof(double));
    v->bv = calloc(layer->m, sizeof(double));
    v->ws = malloc(layer->m * sizeof(double*));
    for (int i = 0; i < layer->m; i++)
        v->ws[i] = calloc(layer->n, sizeof(double));
    v->bs = calloc(layer->m, sizeof(double));
}


// SGD

void sgd(AffineLayer *layer, double lr){
    // SGDによってAffineLayerのパラメータを更新する.
    // 正確にはSGDではなく最急降下法になっているかもしれない.
    // パラメータの目安: lr=0.01
    assert(0.0 < lr && lr <= 0.1);
    for (int i = 0; i < layer->m; i++){
        for (int j = 0; j < layer->n; j++)
            layer->w[i][j] -= lr * layer->dw[i][j];
        layer->b[i] -= lr * layer->db[i];
    }
}


// Momentum

void momentum(AffineLayer *layer, Velocities *v, double lr, double beta){
    // MomentumによってAffineLayerのパラメータを更新する.
    // パラメータの目安: lr=0.01, beta=0.9
    assert(0.0 < lr && lr <= 0.1);
    assert(0.0 < beta && beta < 1.0);
    for (int i = 0; i < layer->m; i++){
        for (int j = 0; j < layer->n; j++){
            v->wv[i][j] = beta*v->wv[i][j] + (1.0-beta)*layer->dw[i][j];
            layer->w[i][j] -= lr * v->wv[i][j];
        }
        v->bv[i] = beta*v->bv[i] + (1.0-beta)*layer->db[i];
        layer->b[i] -= lr * v->bv[i];
    }
}


// RMSProp

void rmsprop(AffineLayer *layer, Velocities *v, double lr, double beta, double epsilon){
    // RMSPropによってAffineLayerのパラメータを更新する.
    // パラメータの目安: lr=0.01, beta=0.9, epsilon=1e-7
    assert(0.0 < lr && lr <= 0.1);
    assert(0.0 < beta && beta < 1.0);
    assert(0.0 < epsilon && epsilon <= 0.01);
    for (int i = 0; i < layer->m; i++){
        for (int j = 0; j < layer->n; j++){
            v->wv[i][j] = beta*v->wv[i][j] + (1.0-beta)*layer->dw[i][j]*layer->dw[i][j];
            layer->w[i][j] -= lr * layer->dw[i][j] / sqrt(v->wv[i][j]+epsilon);
        }
        v->bv[i] = beta*v->bv[i] + (1.0-beta)*layer->db[i]*layer->db[i];
        layer->b[i] -= lr * layer->db[i] / sqrt(v->bv[i]+epsilon);
    }
}


// Adam

void adam(AffineLayer *layer, Velocities *v, double lr, double beta1, double beta2, double epsilon){
    // AdamによってAffineLayerのパラメータを更新する.
    // パラメータの目安: lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7
    assert(0.0 < lr && lr <= 0.1);
    assert(0.0 < beta1 && beta1 < 1.0);
    assert(0.0 < beta2 && beta2 < 1.0);
    assert(0.0 < epsilon && epsilon <= 0.01);
    for (int i = 0; i < layer->m; i++){
        for (int j = 0; j < layer->n; j++){
            v->wv[i][j] = beta1*v->wv[i][j] + (1.0-beta1)*layer->dw[i][j];
            v->ws[i][j] = beta2*v->ws[i][j] + (1.0-beta2)*layer->dw[i][j]*layer->dw[i][j];
            layer->w[i][j] -= lr * v->wv[i][j] / sqrt(v->ws[i][j]+epsilon);
        }
        v->bv[i] = beta1*v->bv[i] + (1.0-beta1)*layer->db[i];
        v->bs[i] = beta2*v->bs[i] + (1.0-beta2)*layer->db[i]*layer->db[i];
        layer->b[i] -= lr * v->bv[i] / sqrt(v->bs[i]+epsilon);
    }
}
