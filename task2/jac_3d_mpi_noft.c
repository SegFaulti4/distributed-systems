#include <math.h>
#include <stdio.h>
#include <mpi.h>

#define DEBUG 1

#define FIRST_SYNC_TAG 1215
#define SECOND_SYNC_TAG 1216
#define FINISH_TAG 1218

#define N 34
#define MAX_ITERATIONS 100

#define Max(a,b) ((a)>(b)?(a):(b))
#define debug_m_printf if (DEBUG && !rank) printf
#define debug_printf if (DEBUG) printf


void sync_edges();

void matrix_init();
void compute();
void relax();
void resid();
void show_result();


int rank, size;
int start_row, last_row;

double A[N][N][N], B[N][N][N];
double eps;


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    debug_m_printf("size: %d\n", size);

    int num_rows = (N - 2) / size;
    start_row = num_rows * rank + 1;
    last_row = start_row + num_rows - 1;
    last_row += (rank == size - 1) ? (N - 2) % size : 0;
    debug_printf("rank: %d, startrow: %d, lastrow: %d\n", rank, start_row, last_row);

    matrix_init();

    for (int it_num = 0; it_num < MAX_ITERATIONS; it_num++) {
        compute();
        sync_edges();
    }

    if (rank == 0) {
        debug_printf("Receiving data\n");
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        for (int i = 1; i < size; i++) {
            MPI_Status status;
            double local_eps;
            MPI_Recv(&start_row, 1, MPI_INT, i, FINISH_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&last_row, 1, MPI_INT, i, FINISH_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&A[start_row][0][0], (last_row - start_row + 1) * N * N, MPI_DOUBLE, i, FINISH_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&local_eps, 1, MPI_DOUBLE, i, FINISH_TAG, MPI_COMM_WORLD, &status);
            eps = Max(eps, local_eps);
        }
        start_row = 0;
        last_row = N - 1;

        MPI_Barrier(MPI_COMM_WORLD);
        debug_printf("Finished\n");
        show_result();
    } else {
        debug_printf("Sending data from %d\n", rank);
        MPI_Send(&start_row, 1, MPI_INT, 0, FINISH_TAG, MPI_COMM_WORLD);
        MPI_Send(&last_row, 1, MPI_INT, 0, FINISH_TAG, MPI_COMM_WORLD);
        MPI_Send(&A[start_row][0][0], (last_row - start_row + 1) * N * N, MPI_DOUBLE, 0, FINISH_TAG, MPI_COMM_WORLD);
        MPI_Send(&eps, 1, MPI_DOUBLE, 0, FINISH_TAG, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

void matrix_init() {
    for (int i = start_row - 1; i <= last_row + 1; i++) {
        for (int j = 0; j <= N - 1; j++) {
            for (int k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
                    A[i][j][k] = 0.;
                } else {
                    A[i][j][k] = (4. + i + j + k);
                }
            }
        }
    }
}

void compute() {
    relax();
    resid();
}

void relax() {
    debug_m_printf("Started relax\n");
    for (int i = start_row; i <= last_row; i++) {
        for (int j = 1; j <= N - 2; j++) {
            for (int k = 1; k <= N - 2; k++) {
                B[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                              A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6.;
            }
        }
    }
}

void resid() {
    int start_flag = start_row == 0 ? 1 : 0;
    int last_flag = last_row == N - 1 ? 1 : 0;

    start_row = start_flag ? start_row + 1 : start_row;
    last_row = last_flag ? last_row - 1 : last_row;

    debug_m_printf("Started resid\n");
    eps = 0.;
    for (int i = start_row; i <= last_row; i++) {
        for (int j = 1; j <= N - 2; j++) {
            for (int k = 1; k <= N - 2; k++) {
                double e;
                e = fabs(A[i][j][k] - B[i][j][k]);
                A[i][j][k] = B[i][j][k];
                eps = Max(eps, e);
            }
        }
    }
    debug_m_printf("Resid eps: %lf\n", eps);

    start_row = start_flag ? start_row - 1 : start_row;
    last_row = last_flag ? last_row + 1 : last_row;
}

void show_result() {
    double s = 0.0;
    for (int i = start_row; i <= last_row; i++) {
        for (int j = 0; j <= N - 1; j++) {
            for (int k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }
        }
    }

    printf("S = %lf\neps = %lf\n", s, eps);

    FILE * res = fopen("result_noft.txt", "w");
    for (int i = start_row; i <= last_row; i++) {
        for (int j = 0; j <= N - 1; j++) {
            for (int k = 0; k <= N - 1; k++) {
                fprintf(res, "%lf ", A[i][j][k]);
            }
            fprintf(res, "\n");
        }
        fprintf(res, "\n");
    }
    fclose(res);
}

void sync_edges() {
    MPI_Request request[4];
    MPI_Status status[4];

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank) {
        MPI_Irecv(&A[start_row - 1][0][0], N * N, MPI_DOUBLE, rank - 1, FIRST_SYNC_TAG, MPI_COMM_WORLD, &request[0]);
        MPI_Isend(&A[start_row][0][0], N * N, MPI_DOUBLE, rank - 1, SECOND_SYNC_TAG, MPI_COMM_WORLD, &request[1]);
    }
    if (rank != size - 1) {
        MPI_Isend(&A[last_row][0][0], N * N, MPI_DOUBLE, rank + 1, FIRST_SYNC_TAG, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&A[last_row + 1][0][0], N * N, MPI_DOUBLE, rank + 1, SECOND_SYNC_TAG, MPI_COMM_WORLD, &request[3]);
    }

    int ll = 4, shift = 0;
    if (!rank) {
        ll -= 2;
        shift = 2;
    }
    if (rank == size - 1) {
        ll -= 2;
    }
    if (ll) {
        MPI_Waitall(ll, &request[shift], status);
    }
}
