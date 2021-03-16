#pragma once
#include <bitset>
#include <cstdio>
#include <vector>
using namespace std;

#define always_inline __inline__ __attribute__((always_inline))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

typedef unsigned int u32;
typedef unsigned long long u64;

struct csr_matrix {
  int m, rows, cols;
  double *nnz, *x, *y, *ans;
  int *col, *rowb, *rowe;
  int *tstart;
  int *tend;
};

void input_matrix(csr_matrix &mat) {
  FILE *fp1 = fopen("nnz.txt", "r");
  FILE *fp2 = fopen("col.txt", "r");
  FILE *fp3 = fopen("rowb.txt", "r");
  FILE *fp4 = fopen("rowe.txt", "r");
  FILE *fp5 = fopen("x.txt", "r");
  FILE *fp6 = fopen("ans.txt", "r");

  mat.nnz = (double *)mkl_malloc(mat.m * sizeof(double), 64);
  mat.col = (int *)mkl_malloc(mat.m * sizeof(int), 64);
  mat.rowb = (int *)mkl_malloc(mat.rows * sizeof(int), 64);
  mat.rowe = (int *)mkl_malloc(mat.rows * sizeof(int), 64);
  mat.x = (double *)mkl_malloc(mat.cols * sizeof(double), 64);
  mat.y = (double *)mkl_malloc(mat.rows * sizeof(double), 64);
  mat.ans = (double *)mkl_malloc(mat.rows * sizeof(double), 64);

  for (int i = 0; i < mat.m; i++) {
    fscanf(fp1, "%lf", &mat.nnz[i]);
  }
  for (int i = 0; i < mat.m; i++) {
    fscanf(fp2, "%d", &mat.col[i]);
  }
  for (int i = 0; i < mat.rows; i++) {
    fscanf(fp3, "%d", &mat.rowb[i]);
  }
  for (int i = 0; i < mat.rows; i++) {
    fscanf(fp4, "%d", &mat.rowe[i]);
  }
  for (int i = 0; i < mat.cols; i++) {
    fscanf(fp5, "%lf", &mat.x[i]);
  }
  for (int i = 0; i < mat.rows; i++) {
    mat.y[i] = 0.0;
  }
  for (int i = 0; i < mat.rows; i++) {
    fscanf(fp6, "%lf", &mat.ans[i]);
  }

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
  fclose(fp4);
  fclose(fp5);
  fclose(fp6);
}

void destroy_matrix(csr_matrix &mat) {
  mkl_free(mat.nnz);
  mkl_free(mat.col);
  mkl_free(mat.rowb);
  mkl_free(mat.rowe);
  mkl_free(mat.x);
  mkl_free(mat.y);
  mkl_free(mat.ans);
}

bool check_answer(csr_matrix &mat) {
  int bad_count = 0;
  for (int i = 0; i < mat.rows; i++) {
    double yi = mat.y[i];
    double ansi = mat.ans[i];
    if (abs(yi - ansi) > 0.01 * abs(ansi) && !(abs(yi) <= 1e-5 && abs(ansi) <= 1e-5)) {  
    	if (bad_count < 10)
        fprintf(stderr, "y[%d] expected %lf got %lf\n", i, mat.ans[i], mat.y[i]);
      bad_count++;
    }
  }
  if (bad_count)
    fprintf(stderr, "bad_count: %d\n", bad_count);
  return bad_count == 0 ? true : false;
}

csr_matrix apply_order(csr_matrix &mat, vector<vector<int>> &tasks, int copy_oob = true) {
  csr_matrix ret;

  ret.m = mat.m;
  ret.rows = mat.rows;
  ret.cols = mat.cols;
  ret.nnz = (double *)mkl_malloc(mat.m * sizeof(double), 64);
  ret.col = (int *)mkl_malloc(mat.m * sizeof(int), 64);
  ret.rowb = (int *)mkl_malloc(mat.rows * sizeof(int), 64);
  ret.rowe = (int *)mkl_malloc(mat.rows * sizeof(int), 64);
  if (copy_oob) {
    ret.x = (double *)mkl_malloc(mat.cols * sizeof(double), 64);
    ret.y = (double *)mkl_malloc(mat.rows * sizeof(double), 64);
    ret.ans = (double *)mkl_malloc(mat.rows * sizeof(double), 64);
  }
  ret.tstart = (int *)mkl_malloc(tasks.size() * sizeof(int), 64);
  ret.tend = (int *)mkl_malloc(tasks.size() * sizeof(int), 64);

  if (copy_oob) {
    for (int i = 0; i < mat.cols; i++)
      ret.x[i] = mat.x[i];

    for (int i = 0; i < mat.rows; i++)
      ret.y[i] = 0;

    for (int i = 0; i < mat.rows; i++)
      ret.ans[i] = mat.ans[i];
  }

  int npos = 0, pos = 0;
  int start = 0, t = 0;
  for (vector<int> &task : tasks) {
    ret.tstart[t] = start;
    ret.tend[t++] = start + task.size();
    start += task.size();
    for (int row : task) {
      int b = mat.rowb[row];
      int e = mat.rowe[row];
      ret.rowb[pos] = npos;
      ret.rowe[pos++] = npos + e - b;
      for (int i = b; i < e; i++) {
        ret.nnz[npos] = mat.nnz[i];
        ret.col[npos++] = mat.col[i];
      }
    }
  }

  return ret;
}

