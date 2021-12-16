#include "neural_network.c"

#include <time.h>

#define TRAIN_DATA "mnist_train_49.txt"
#define TEST_DATA "mnist_test_49.txt"
#define IMAGE_SIZE 28
#define TRAIN_SIZE 12665
#define TEST_SIZE 2115


/*
MNISTの画像分類を行う.
簡単のため, 4と9の分類に限定する.
*/


double stop_watch(struct timespec start_time, struct timespec end_time) {
    // 経過時間を秒単位で返す.
    /* 使用例
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    // 2秒間処理を止める.
    sleep(2);
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("%lf\n", stop_watch(start_time, end_time));
    */
    long int sec = end_time.tv_sec - start_time.tv_sec;
    long int nsec = end_time.tv_nsec - start_time.tv_nsec;
    return (double) sec + (double) nsec / pow(10, 9);
}


int main(void){

    // モデルの準備
    NeuralNetwork *nn;
    nn = malloc(sizeof(NeuralNetwork));
    int model_size[4] = {IMAGE_SIZE*IMAGE_SIZE, 32, 32, 1};
    nn_init(nn, model_size);
    
    // データセットの準備
    FILE *fp;

    double **X_train, **y_train;
    X_train = malloc(TRAIN_SIZE * sizeof(double*));
    y_train = malloc(TRAIN_SIZE * sizeof(double*));
    for (int i = 0; i < TRAIN_SIZE; i++){
        X_train[i] = malloc(IMAGE_SIZE*IMAGE_SIZE * sizeof(double));
        y_train[i] = malloc(1 * sizeof(double));
    }
    fp = fopen(TRAIN_DATA, "r");
    for (int i = 0; i < TRAIN_SIZE; i++){
        fscanf(fp, "%lf", &y_train[i][0]);
        if (y_train[i][0] == 4.0)
            y_train[i][0] = 0.0;
        else
            y_train[i][0] = 1.0;
        for (int j = 0; j < IMAGE_SIZE*IMAGE_SIZE; j++)
            fscanf(fp, "%lf", &X_train[i][j]);
    }
    fclose(fp);
    
    double **X_test, **y_test;
    X_test = malloc(TEST_SIZE * sizeof(double*));
    y_test = malloc(TEST_SIZE * sizeof(double*));
    for (int i = 0; i < TEST_SIZE; i++){
        X_test[i] = malloc(IMAGE_SIZE*IMAGE_SIZE * sizeof(double));
        y_test[i] = malloc(1 * sizeof(double));
    }
    fp = fopen(TEST_DATA, "r");
    for (int i = 0; i < TEST_SIZE; i++){
        fscanf(fp, "%lf", &y_test[i][0]);
        if (y_test[i][0] == 4.0)
            y_test[i][0] = 0.0;
        else
            y_test[i][0] = 1.0;
        for (int j = 0; j < IMAGE_SIZE*IMAGE_SIZE; j++)
            fscanf(fp, "%lf", &X_test[i][j]);
    }
    fclose(fp);

    // 学習

    // 時間計測
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_REALTIME, &start_time);

    // モデルを学習させる.
    double lr = 0.001;
    nn_fit(nn, X_train, y_train, TRAIN_SIZE, X_test, y_test, TEST_SIZE, lr, 5);

    // 時間計測
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("%lf\n", stop_watch(start_time, end_time));

    return 0;
}