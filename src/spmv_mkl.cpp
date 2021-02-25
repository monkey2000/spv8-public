#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mkl.h"
#include "mkl_spblas.h"
//#include "tbb/task_scheduler_init.h"
#define __USE_GNU
#include "sched.h"
#include "unistd.h"

// #define LOOP_COUNT 1000
int main(int argc, char *argv[]) {
  double *nnz, *x, *y;
  int *col, *rowb, *rowe;
  int m, n, c, i, r, loop_cnt;
  double alpha, beta;
  double duration;
  double start, end;
  sparse_matrix_t A;
  struct matrix_descr tt = {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER,
                            SPARSE_DIAG_NON_UNIT};
  sparse_status_t stat;
  int flag = 1;
  int thread_num = 8;

  FILE *fp1 = fopen("nnz.txt", "r");
  FILE *fp2 = fopen("col.txt", "r");
  FILE *fp3 = fopen("rowb.txt", "r");
  FILE *fp4 = fopen("rowe.txt", "r");
  FILE *fp5 = fopen("x.txt", "r");
  m = 924886, n = 194085,
  c = 194085;  // m is number of non-zeros, n is matrix row, c is matrix column
  loop_cnt = 5000;

  if (access("info.txt", F_OK) != -1) {
    FILE *info = fopen("info.txt", "r");
    fscanf(info, "%d", &m);
    fscanf(info, "%d", &n);
    fscanf(info, "%d", &c);
    fclose(info);
  }
 
  if (argc > 3) {
    sscanf(argv[1], "%d", &loop_cnt);
    sscanf(argv[2], "%d", &flag);
    sscanf(argv[3], "%d", &thread_num);
  }

  alpha = 1.0;
  beta = 0.0;
  nnz = (double *)mkl_malloc(m * sizeof(double), 64);
  col = (int *)mkl_malloc(m * sizeof(int), 64);
  rowb = (int *)mkl_malloc(n * sizeof(int), 64);
  rowe = (int *)mkl_malloc(n * sizeof(int), 64);
  x = (double *)mkl_malloc(c * sizeof(double), 64);
  y = (double *)mkl_malloc(n * sizeof(double), 64);

  for (i = 0; i < m; i++) {
    fscanf(fp1, "%lf", &nnz[i]);
  }
  for (i = 0; i < m; i++) {
    fscanf(fp2, "%d", &col[i]);
  }
  for (i = 0; i < n; i++) {
    fscanf(fp3, "%d", &rowb[i]);
  }
  for (i = 0; i < n; i++) {
    fscanf(fp4, "%d", &rowe[i]);
  }
  for (i = 0; i < c; i++) {
    fscanf(fp5, "%lf", &x[i]);
  }
  for (i = 0; i < n; i++) {
    y[i] = 0.0;
  }

  // tbb::task_scheduler_init init(thread_num);
  mkl_set_num_threads(thread_num);
  mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, c, rowb, rowe, col,
                          nnz);

  if (flag) {
    start = dsecnd();

    mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, tt, 200);
    mkl_sparse_optimize(A);

    end = dsecnd();
    duration = (double)(end - start);
  }

  for (r = 0; r < 300; r++) {
    stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, tt, x,
                           beta, y);
  }

  start = dsecnd();
  for (r = 0; r < loop_cnt; r++) {
    stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, tt, x,
                           beta, y);
  }
  end = dsecnd();
  duration = (double)(end - start) / loop_cnt;
  printf("%lf\n", duration * 1000);

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
  fclose(fp4);
  fclose(fp5);
  mkl_free(nnz);
  mkl_free(col);
  mkl_free(rowb);
  mkl_free(rowe);
  mkl_free(x);
  mkl_free(y);

  return 0;
}
