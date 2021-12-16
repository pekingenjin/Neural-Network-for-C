#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define LEN_ARRAY(a) (sizeof(a) / sizeof(a[0]))
#define SWAP(a, b) (a ^= b, b = a^b, a ^= b)


/*
ニューラルネットワークで使用する, 数学的な関数を実装した.
*/


// ------ loss functions ------

double sse(const double y[], const double t[], double dout[], int len){
    /*
    Sum of Squared Error
    モデルの出力yと正解tの二乗和誤差を計算する.
    doutに勾配を代入し, 二乗和誤差を出力する.
    */
    double res = 0.0;
    for (int i = 0; i < len; i++){
        dout[i] = y[i] - t[i];
        res += (y[i]-t[i]) * (y[i]-t[i]);
    }
    return res / 2.0;
}


// ------ random functions ------

void shuffle(int array[], int len){
    for (int i = 0; i < len; i++){
        int j = rand() % len;
        SWAP(array[i], array[j]);
    }
}

double uniform(double left, double right){
    // left以上right以下の一様乱数を返す.
    assert(left <= right);
    return left + (right-left)*(double)rand()/(double)RAND_MAX;
}

double rand_normal(double mu, double sigma){
    // 平均mu, 標準偏差sigmaの正規分布に従う乱数を返す.
    // Box-Muller's method
    return mu + sigma * sqrt(-2.0*log(uniform(0.0,1.0))) * cos(2.0*M_PI*uniform(0.0,1.0));
}

double xavier(int n){
    /*
    Xavierの初期値を返す.
    sigmoid関数やtanhと相性がよい.
    */
    return rand_normal(0.0, 1.0/sqrt(n));
}

double he(int n){
    /*
    Heの初期値を返す.
    ReLU関数と相性がよい.
    */
    return rand_normal(0.0, sqrt(2.0/(double)n));
}


// ------ matrix and vector ------

void matrix_init(double **m, int h, int w){
    // h行w列の2次元配列を初期化する.
    for (int i = 0; i < h; i++){
        m[i] = malloc(w * sizeof(double));
    }
}

void delete_matrix(double **m, int h){
    // 2次元配列のメモリを解放する.
    for (int i = 0; i < h; i++)
        free(m[i]);
    free(m);
}

void vec_add_vec(const double a[], const double b[], double c[], int len){
    // a+bをcに代入する.
    for (int i = 0; i < len; i++)
        c[i] = a[i] + b[i];
}

double vec_mul_vec(const double a[], const double b[], int len){
    // aとbの内積を返す.
    double res = 0.0;
    for (int i = 0; i < len; i++)
        res += a[i] * b[i];
    return res;
}

void vec_mul_vecT(const double a[], const double b[], double **res, int h, int w){
    // a*b^Tをresに代入する.
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            res[i][j] = a[i] * b[j];
    }
}

void mat_mul_vec(double **m, const double v[], double res[], int h, int w){
    // m*vをresに代入する.
    for (int i = 0; i < h; i++)
        res[i] = vec_mul_vec(m[i], v, w);
}

double **transpose(double **m, int h, int w){
    // mの転置行列を返す.
    double **mt = malloc(w * sizeof(double*));
    matrix_init(mt, w, h);
    // 値を代入する.
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            mt[j][i] = m[i][j];
    }
    return mt;
}
