#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

#include <bits/stdc++.h>
//#include <atcoder/all>
using namespace std;
//using namespace atcoder;

// Reference: https://github.com/oreilly-japan/deep-learning-from-scratch/tree/master/common

// ------ random functions ------

double uniform(double l, double r) {
    /* l以上r以下の実数を返す. */
    assert(l <= r);
    return l + (r-l) * (double)rand() / RAND_MAX;
}

double rand_normal(double mu, double sigma) {
    /* 平均mu, 標準偏差sigmaの正規分布に従う乱数を返す.
    Box-Muller's method */
    assert(0.0 <= sigma);
    return mu + sigma * sqrt(-2.0*log(uniform(0.0,1.0))) * cos(2.0*M_PI*uniform(0.0,1.0));
}

double xavier(int n) {
    /* Xavierの初期値を返す.
    sigmoid関数やtanhと相性がよい. */
    assert(0 < n);
    return rand_normal(0.0, 1.0/sqrt(n));
}

double he(int n) {
    /* Heの初期値を返す.
    ReLU関数と相性がよい. */
    assert(0 < n);
    return rand_normal(0.0, sqrt(2.0/(double)n));
}

// ------ matrix calculation ------

vector<vector<double>> transpose(const vector<vector<double>>& a) {
    /* aの転置行列を返す. */
    assert(!a.empty() && !a[0].empty());
    vector<vector<double>> ret(a[0].size(), vector<double>(a.size()));
    for (int i = 0; i < (int)a.size(); i++) {
        for (int j = 0; j < (int)a[0].size(); j++) {
            ret[j][i] = a[i][j];
        }
    }
    return ret;
}

vector<vector<double>> dot(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    assert(!a.empty() && !a[0].empty() && !b.empty() && !b[0].empty());
    assert(a[0].size() == b.size());
    vector<vector<double>> ret(a.size(), vector<double>(b[0].size(),0.0));
    vector<vector<double>> tb = transpose(b);
    for (int i = 0; i < (int)a.size(); i++) {
        for (int j = 0; j < (int)tb.size(); j++) {
            for (int k = 0; k < (int)a[0].size(); k++) {
                ret[i][j] += a[i][k] * tb[j][k];
            }
        }
    }
    return ret;
}

// ------ optimizers ------

struct Optimizer {
    /* abstract class */
    virtual void update(vector<double>& params, vector<double>& grads) = 0;
};

struct SGD : Optimizer {
    /* 確率的勾配降下法ではなく, 最急降下法の実装になっている. */
    double lr;

    SGD(double learning_rate=0.01) {
        assert(0.0 < learning_rate && learning_rate < 1.0);
        lr = learning_rate;
    }

    void update(vector<double>& params, vector<double>& grads) override {
        assert(!params.empty() && !grads.empty());
        assert(params.size() == grads.size());
        for (int i = 0; i < (int)params.size(); i++) {
            params[i] -= lr * grads[i];
        }
    }
};

struct Momentum : Optimizer {
    double lr;
    double beta;
    vector<double> v;

    Momentum(double learning_rate=0.01, double momentum=0.9) {
        assert(0.0 < learning_rate && learning_rate < 1.0);
        assert(0.0 < momentum && momentum < 1.0);
        lr = learning_rate;
        beta = momentum;
    }

    void update(vector<double>& params, vector<double>& grads) override {
        assert(!params.empty() && !grads.empty());
        assert(params.size() == grads.size());
        if (v.empty()) {
            v = vector<double>(params.size(), 0.0);
        }
        for (int i = 0; i < (int)params.size(); i++) {
            v[i] = beta*v[i] - (1.0-beta)*grads[i];
            params[i] -= lr * v[i];
        }
    }
};

struct RMSprop : Optimizer {
    double lr;
    double beta;
    double epsilon;
    vector<double> v;

    RMSprop(double learning_rate=0.01, double decay_rate=0.99, double delta=1e-7) {
        assert(0.0 < learning_rate && learning_rate < 1.0);
        assert(0.0 < decay_rate && decay_rate < 1.0);
        assert(delta <= 0.1);
        lr = learning_rate;
        beta = decay_rate;
        epsilon = delta;
    }

    void update(vector<double>& params, vector<double>& grads) override {
        assert(!params.empty() && !grads.empty());
        assert(params.size() == grads.size());
        if (v.empty()) {
            v = vector<double>(params.size(), 0.0);
        }
        for (int i = 0; i < (int)params.size(); i++) {
            v[i] = beta*v[i] + (1.0-beta)*grads[i]*grads[i];
            params[i] -= lr * grads[i] / sqrt(v[i]+epsilon);
        }
    }
};

struct Adam : Optimizer {
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    vector<double> v;
    vector<double> s;

    Adam(double learning_rate=0.01, double momentum=0.9, double decay_rate=0.999, double delta=1e-7) {
        assert(0.0 < learning_rate && learning_rate < 1.0);
        assert(0.0 < momentum && momentum < 1.0);
        assert(0.0 < decay_rate && decay_rate < 1.0);
        assert(delta <= 0.1);
        lr = learning_rate;
        beta1 = momentum;
        beta2 = decay_rate;
        epsilon = delta;
    }

    void update(vector<double>& params, vector<double>& grads) override {
        assert(!params.empty() && !grads.empty());
        assert(params.size() == grads.size());
        if (v.empty()) {
            v = vector<double>(params.size(), 0.0);
            s = vector<double>(params.size(), 0.0);
        }
        for (int i = 0; i < (int)params.size(); i++) {
            v[i] = beta1*v[i] + (1.0-beta1)*grads[i];
            s[i] = beta2*s[i] + (1.0-beta2)*grads[i]*grads[i];
            params[i] -= lr * v[i] / sqrt(s[i]+epsilon);
        }
    }
};

// ------ layers ------

struct Layer {
    /* abstract class */
    virtual vector<vector<double>> forward(const vector<vector<double>>& x) = 0;
    virtual vector<vector<double>> backward(vector<vector<double>>& dout) = 0;
    virtual void update() = 0;
};

struct ReluLayer : Layer {
    vector<vector<bool>> mask;

    vector<vector<double>> forward(const vector<vector<double>>& x) override {
        assert(!x.empty() && !x[0].empty());
        vector<vector<double>> ret(x.size(), vector<double>(x[0].size(), 0.0));
        mask = vector<vector<bool>>(x.size(), vector<bool>(x[0].size(), true));
        for (int i = 0; i < (int)x.size(); i++) {
            for (int j = 0; j < (int)x[0].size(); j++) {
                if (0 < x[i][j]) {
                    ret[i][j] = x[i][j];
                    mask[i][j] = false;
                }
            }
        }
        return ret;
    }

    vector<vector<double>> backward(vector<vector<double>>& dout) override {
        assert(!dout.empty() && !dout[0].empty() && !mask.empty() && !mask[0].empty());
        assert(dout.size() == mask.size() && dout[0].size() == mask[0].size());
        for (int i = 0; i < (int)dout.size(); i++) {
            for (int j = 0; j < (int)dout[0].size(); j++) {
                if (mask[i][j]) {
                    dout[i][j] = 0.0;
                }
            }
        }
        return dout;
    }

    void update() override {}
};

struct SigmoidLayer : Layer {
    vector<vector<double>> out;

    vector<vector<double>> forward(const vector<vector<double>>& x) override {
        assert(!x.empty() && !x[0].empty());
        out = vector<vector<double>>(x.size(), vector<double>(x[0].size()));
        for (int i = 0; i < (int)x.size(); i++) {
            for (int j = 0; j < (int)x[0].size(); j++) {
                out[i][j] = 1.0 / (1.0 + exp(-x[i][j]));
            }
        }
        return out;
    }

    vector<vector<double>> backward(vector<vector<double>>& dout) override {
        assert(!dout.empty() && !dout[0].empty() && !out.empty() && !out[0].empty());
        assert(dout.size() == out.size() && dout[0].size() == out[0].size());
        for (int i = 0; i < (int)dout.size(); i++) {
            for (int j = 0; j < (int)dout[0].size(); j++) {
                dout[i][j] *= (1.0 - out[i][j]) * out[i][j];
            }
        }
        return dout;
    }

    void update() override {}
};

struct AffineLayer : Layer {
    int n;
    int m;
    vector<vector<double>> w;
    vector<double> b;
    vector<vector<double>> input;
    vector<vector<double>> dw;
    vector<double> db;
    vector<Optimizer*> w_opt;
    Optimizer* b_opt;

    AffineLayer(int in, int out, string initial_value="xavier", string optimizer="adam", double lr=0.01) {
        assert(0 < in && 0 < out);
        n = in;
        m = out;
        transform(initial_value.begin(), initial_value.end(), initial_value.begin(), ::tolower);
        w = vector<vector<double>>(n, vector<double>(m));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (initial_value == "xavier") {
                    w[i][j] = xavier(n);
                } else if (initial_value == "he") {
                    w[i][j] = he(n);
                } else {
                    cout << "error: unknown initial_value: " << initial_value << endl;
                    return;
                }
            }
        }
        b = vector<double>(m, 0.0);

        transform(optimizer.begin(), optimizer.end(), optimizer.begin(), ::tolower);
        w_opt = vector<Optimizer*>(n);
        if (optimizer == "sgd") {
            for (int i = 0; i < n; i++) {
                w_opt[i] = new SGD(lr);
            }
            b_opt = new SGD(lr);
        } else if (optimizer == "momentum") {
            for (int i = 0; i < n; i++) {
                w_opt[i] = new Momentum(lr);
            }
            b_opt = new Momentum(lr);
        } else if (optimizer == "rmsprop") {
            for (int i = 0; i < n; i++) {
                w_opt[i] = new RMSprop(lr);
            }
            b_opt = new RMSprop(lr);
        } else if (optimizer == "adam") {
            for (int i = 0; i < n; i++) {
                w_opt[i] = new Adam(lr);
            }
            b_opt = new Adam(lr);
        } else {
            cout << "error: unknown optimizer: " << optimizer << endl;
            return;
        }
    }

    vector<vector<double>> forward(const vector<vector<double>>& x) override {
        assert(!x.empty() && (int)x[0].size() == n);
        input = x;
        vector<vector<double>> out = dot(x, w);
        for (int i = 0; i < (int)out.size(); i++) {
            for (int j = 0; j < m; j++) {
                out[i][j] += b[j];
            }
        }
        return out;
    }

    vector<vector<double>> backward(vector<vector<double>>& dout) override {
        assert(dout.size() == input.size() && (int)dout[0].size() == m);
        vector<vector<double>> dx = dot(dout, transpose(w));
        dw = dot(transpose(input), dout);
        db = vector<double>(m, 0.0);
        for (int i = 0; i < (int)input.size(); i++) {
            for (int j = 0; j < m; j++) {
                db[j] += dout[i][j];
            }
        }
        return dx;
    }
    
    void update() override {
        for (int i = 0; i < n; i++) {
            w_opt[i]->update(w[i], dw[i]);
        }
        b_opt->update(b, db);
    }
};

// ------ loss functions ------

double sse(const vector<double>& y, const vector<double>& t, vector<double>& dout) {
    /* Sum of Squared Error
    モデルの出力yと正解tの二乗和誤差を返す.
    doutに勾配を代入する. */
    assert(y.size() == dout.size() && t.size() == dout.size());
    double ret = 0.0;
    for (int i = 0; i < (int)dout.size(); i++) {
        dout[i] = y[i] - t[i];
        ret += (y[i]-t[i]) * (y[i]-t[i]);
    }
    return ret / 2.0;
}

// ------ neural network ------

struct NeuralNetwork {
    vector<Layer*> layers;
    vector<vector<double>> dout;

    vector<vector<double>> predict(vector<vector<double>> x) {
        for (Layer* layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    vector<double> test(vector<vector<double>>& x, vector<vector<double>>& t) {
        vector<vector<double>> y = predict(x);
        assert(y.size() == t.size() && y[0].size() == t[0].size());

        vector<double> ret(y.size());
        dout = vector<vector<double>>(y.size(), vector<double>(y[0].size()));
        for (int i = 0; i < (int)y.size(); i++) {
            ret[i] = sse(y[i], t[i], dout[i]);
        }
        return ret;
    }

    vector<double> train(vector<vector<double>>& x, vector<vector<double>>& t) {
        vector<double> ret = test(x, t);
        for (int i = layers.size()-1; 0 <= i; i--) {
            dout = layers[i]->backward(dout);
            layers[i]->update();
        }
        return ret;
    }

    void fit(vector<vector<double>>& x_train, vector<vector<double>>& y_train, vector<vector<double>>& x_test, vector<vector<double>>& y_test, int epoch, int batch_size) {
        assert(x_train.size() == y_train.size() && x_test.size() == y_test.size());
        for (int i = 0; i < epoch; i++) {
            cout << "epoch " << i+1 << endl;

            random_device seed_gen;
            mt19937 engine(seed_gen());
            shuffle(x_train.begin(), x_train.end(), engine);
            
            vector<double> losses;
            for (int j = 0; j < x_train.size(); j += batch_size) {
                vector<vector<double>> x;
                vector<vector<double>> y;
                for (int k = j; k < min((int)x_train.size(),j+batch_size); k++) {
                    x.push_back(x_train[k]);
                    y.push_back(y_train[k]);
                }
                vector<double> loss = train(x, y);
                losses.insert(losses.end(), loss.begin(), loss.end());
            }
            cout << "train loss: " << accumulate(losses.begin(), losses.end(), 0.0) / losses.size() << endl;

            losses = test(x_test, y_test);
            cout << "test loss: " << accumulate(losses.begin(), losses.end(), 0.0) / losses.size() << endl;
        }
    }
};

int main() {
    NeuralNetwork nn;
    nn.layers.push_back(new AffineLayer(784, 32, "He"));
    nn.layers.push_back(new ReluLayer);
    nn.layers.push_back(new AffineLayer(32, 32, "He"));
    nn.layers.push_back(new ReluLayer);
    nn.layers.push_back(new AffineLayer(32, 1, "He"));

    vector<vector<double>> x_train(12665, vector<double>(784));
    vector<vector<double>> y_train(12665);
    ifstream train_data("mnist_train_49.txt");
    for (int i = 0; i < (int)x_train.size(); i++) {
        int t;
        train_data >> t;
        if (t == 4) {
            y_train[i] = {0.0};
        } else {
            y_train[i] = {1.0};
        }
        for (int j = 0; j < 784; j++) {
            train_data >> x_train[i][j];
        }
    }
    train_data.close();

    vector<vector<double>> x_test(2115, vector<double>(784));
    vector<vector<double>> y_test(2115);
    ifstream test_data("mnist_test_49.txt");
    for (int i = 0; i < (int)x_test.size(); i++) {
        int t;
        test_data >> t;
        if (t == 4) {
            y_test[i] = {0.0};
        } else {
            y_test[i] = {1.0};
        }
        for (int j = 0; j < 784; j++) {
            test_data >> x_test[i][j];
        }
    }
    test_data.close();

    int batch_size = 32;
    int epoch = 5;
    nn.fit(x_train, y_train, x_test, y_test, epoch, batch_size);
}
