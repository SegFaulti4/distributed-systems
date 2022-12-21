#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <signal.h>
#include <unistd.h>

#define DEBUG 1
#define RECOVERY_PROC_NUM 1
#define NULL_RANK (-1)
#define NULL_WORKER (-1)
#define RECOVERY_IMPOSSIBLE 1
#define RECOVERY_FAILED 2

#define FIRST_SYNC_TAG 1215
#define SECOND_SYNC_TAG 1216
#define RECOVERY_REQ_TAG 1217
#define WORKER_FINISH_TAG 1218

#define N 34
#define MAX_ITERATIONS 100

#define Max(a,b) ((a)>(b)?(a):(b))
#define debug_m_printf if (DEBUG && !rank) printf
#define debug_printf if (DEBUG) printf

#define suicide if (rank == 1 && it_num == 2 && state == PENDING_FIRST_SYNC) { printf("Goodbye...\n"); fflush(stdout); raise(SIGTERM); }

/*
 * PROCESS ENTRY FUNCTIONS
 */

void master_entry();
void recovery_entry();
void worker_entry();

/*
 * WORKER PROCESS FUNCTIONS
 */

void worker_init();
void worker_state();
void worker_sync(int);
void free_workers_info();

/*
 * WORKER RECOVERY FUNCTIONS
 */

void save_worker_checkpoint();
void load_worker_checkpoint(int);
void worker_recovery(int, int);

/*
 * MATRIX OPERATION FUNCTIONS
 */

void matrix_init();
void compute();
void relax();
void resid();
void show_result();


// GENERAL INFO
int rank, size;
int recovery_proc_num, worker_proc_num;

// MASTER INFO
int *worker_rank, *worker_north_rank, *worker_south_rank, *worker_south_first;

// WORKER INFO
int north_rank, south_rank, south_first;
int start_row, last_row, it_num;

typedef enum {
    PENDING_COMPUTE,
    PENDING_FIRST_SYNC,
    PENDING_SECOND_SYNC
} Worker_state;

Worker_state state;

double A[N][N][N], B[N][N][N];
double eps;


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    recovery_proc_num = RECOVERY_PROC_NUM;
    worker_proc_num = size - 1 - recovery_proc_num;
    debug_m_printf("size: %d, recovery processes: %d, worker processes: %d\n", size, recovery_proc_num, worker_proc_num);

    if (worker_proc_num < 1) {
        debug_m_printf("Not enough worker processes - %d\n", worker_proc_num);
        MPI_Finalize();
        return 0;
    }

    worker_rank = malloc(sizeof(*worker_rank) * worker_proc_num);
    worker_north_rank = malloc(sizeof(*worker_north_rank) * worker_proc_num);
    worker_south_rank = malloc(sizeof(*worker_south_rank) * worker_proc_num);
    worker_south_first = malloc(sizeof(*worker_south_first) * worker_proc_num);

    for (int i = 0; i < worker_proc_num; i++) {
        int worker_r = i + 1;
        worker_rank[i] = worker_r;
        worker_north_rank[i] = worker_r == 1 ? NULL_RANK : worker_r - 1;
        worker_south_rank[i] = worker_r == worker_proc_num ? NULL_RANK : worker_r + 1;
        worker_south_first[i] = worker_r & 1;
        debug_m_printf("worker num: %d, rank: %d, north: %d, south: %d, south_first: %d\n",
                       i, worker_r, worker_north_rank[i], worker_south_rank[i], worker_south_first[i]);
    }

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        master_entry();
    } else if (rank > worker_proc_num) {
        recovery_entry();
    } else {
        worker_entry();
    }

    MPI_Finalize();
    return 0;
}


/***************************
 * PROCESS ENTRY FUNCTIONS *
 ***************************/

void master_entry() {
    /*
     * Задача процесса-мастера - следить за ходом исполнения рабочих процессов
     * Процесс-мастер обрабатывает ситуации падения рабочих процессов
     * На место упавшего рабочего процесса встает один из резервных
     */

    int unfinished_workers = worker_proc_num;
    int test[worker_proc_num];
    MPI_Request test_request[worker_proc_num];
    MPI_Status test_status;

    // Non-blocking receives from all workers
    for (int i = 0; i < worker_proc_num; i++) {
        MPI_Irecv(&test[i], 1, MPI_INT, worker_rank[i], WORKER_FINISH_TAG, MPI_COMM_WORLD, &test_request[i]);
    }

    while (unfinished_workers) {
        debug_printf("From %d (master) - unfinished workers: %d\n", rank, unfinished_workers);
        int idx = -1;

        // This operation completes if any worker is dead or finished
        MPI_Waitany(worker_proc_num, test_request, &idx, &test_status);
        debug_printf("From %d (master) - got message from %d: SOURCE: %d, TAG: %d, ERROR: %d\n",
                     rank, worker_rank[idx], test_status.MPI_SOURCE, test_status.MPI_TAG, test_status.MPI_ERROR);

        if (test_status.MPI_SOURCE != worker_rank[idx]) {
            // Found dead worker - try to recover
            int dead_rank = worker_rank[idx];
            debug_printf("From %d (master) - found dead process %d\n", rank, dead_rank);

            if (recovery_proc_num == 0) {
                printf("From %d (master) - no recovery processes, abort", rank);
                MPI_Abort(MPI_COMM_WORLD, RECOVERY_IMPOSSIBLE);
                return;
            }

            int recovery_proc_rank = size - recovery_proc_num;
            debug_printf("From %d (master) - recovering dead process with %d\n", rank, recovery_proc_rank);
            worker_rank[idx] = recovery_proc_rank;

            debug_printf("From %d (master) - sending recovery data to %d\n", rank, recovery_proc_rank);
            int err, tmp;
            MPI_Status tmp_status;
            err = MPI_Send(&dead_rank, 1, MPI_INT, recovery_proc_rank, RECOVERY_REQ_TAG, MPI_COMM_WORLD);
            if (!err) err = MPI_Send(&worker_north_rank[idx], 1, MPI_INT, recovery_proc_rank, RECOVERY_REQ_TAG, MPI_COMM_WORLD);
            if (!err) err = MPI_Send(&worker_south_rank[idx], 1, MPI_INT, recovery_proc_rank, RECOVERY_REQ_TAG, MPI_COMM_WORLD);
            if (!err) err = MPI_Send(&worker_south_first[idx], 1, MPI_INT, recovery_proc_rank, RECOVERY_REQ_TAG, MPI_COMM_WORLD);
            if (!err) err = MPI_Recv(&tmp, 1, MPI_INT, recovery_proc_rank, RECOVERY_REQ_TAG, MPI_COMM_WORLD, &tmp_status);

            if (err) {
                printf("From %d (master) - failed to recover process\n", rank);
                MPI_Abort(MPI_COMM_WORLD, RECOVERY_FAILED);
                return;
            }
            recovery_proc_num--;
            debug_printf("From %d (master) - successful recovery\n", rank);

            MPI_Request send_req[2];
            MPI_Status send_status[2];
            int count = 0;

            // Send recovered process rank to neighbours
            debug_printf("From %d (master) - sending data to %d and %d\n",
                         rank, worker_north_rank[idx], worker_south_rank[idx]);
            if (worker_north_rank[idx] != NULL_WORKER) {
                MPI_Isend(&recovery_proc_rank, 1, MPI_INT, worker_north_rank[idx], RECOVERY_REQ_TAG, MPI_COMM_WORLD, &send_req[count]);
                worker_south_rank[worker_north_rank[idx]] = recovery_proc_rank;
                count++;
            }
            if (worker_south_rank[idx] != NULL_WORKER) {
                MPI_Isend(&recovery_proc_rank, 1, MPI_INT, worker_south_rank[idx], RECOVERY_REQ_TAG, MPI_COMM_WORLD, &send_req[count]);
                worker_north_rank[worker_south_rank[idx]] = recovery_proc_rank;
                count++;
            }

            // Wait for neighbours to receive recovered process rank
            MPI_Waitall(count, send_req, send_status);
            if ((count >= 1 && send_status[0].MPI_ERROR) || (count >= 2 && send_status[1].MPI_ERROR)) {
                printf("From %d (master) - failed to send recovered rank\n", rank);
                MPI_Abort(MPI_COMM_WORLD, RECOVERY_FAILED);
                return;
            }
            debug_printf("From %d (master) - finished sending\n", rank);

            // Non-blocking receive from new process
            MPI_Irecv(&test[idx], 1, MPI_INT, worker_rank[idx], WORKER_FINISH_TAG, MPI_COMM_WORLD, &test_request[idx]);
        } else {
            // Process is not dead - it finished computing and now is waiting to send its data
            printf("From %d (master) - found finished process %d\n", rank, worker_rank[idx]);
            unfinished_workers--;
        }
    }
    debug_printf("From %d (master) - all workers finished\n", rank);

    debug_printf("From %d (master) - stopping recovery processes\n", rank);
    for (int i = 1; i <= recovery_proc_num; i++) {
        int tmp = NULL_RANK;
        debug_printf("From %d (master) - stop recovery proc %d\n", rank, size - i);
        MPI_Send(&tmp, 1, MPI_INT, size - i, RECOVERY_REQ_TAG, MPI_COMM_WORLD);
    }

    debug_printf("From %d (master) - receiving data from workers\n", rank);
    eps = 0.;
    for (int i = 0; i < worker_proc_num; i++) {
        MPI_Status status;
        double local_eps;
        debug_printf("From %d (master) - receiving data from %d\n", rank, worker_rank[i]);
        MPI_Recv(&start_row, 1, MPI_INT, worker_rank[i], WORKER_FINISH_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&last_row, 1, MPI_INT, worker_rank[i], WORKER_FINISH_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&A[start_row][0][0], (last_row - start_row + 1) * N * N, MPI_DOUBLE, worker_rank[i], WORKER_FINISH_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&local_eps, 1, MPI_DOUBLE, worker_rank[i], WORKER_FINISH_TAG, MPI_COMM_WORLD, &status);
        eps = Max(eps, local_eps);
        debug_printf("From %d (master) - received data\n", rank);
    }
    start_row = 0;
    last_row = N - 1;

    debug_printf("From %d (master) - finished\n", rank);
    show_result();
}

void recovery_entry() {
    /*
     * Задача восстанавливающего процесса - дождаться запроса от процесса-мастера
     * и встать на место упавшего рабочего процесса,
     * либо завершить свою работу по запросу мастера
     */
    MPI_Status status;
    int dead_rank;
    debug_printf("From %d (recovery) - waiting request\n", rank);
    MPI_Recv(&dead_rank, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD, &status);
    if (dead_rank == NULL_RANK) {
        debug_printf("From %d (recovery) - gracefully stopping\n", rank);
        return;
    }
    debug_printf("From %d (recovery) - got request to recover %d\n", rank, dead_rank);

    MPI_Recv(&north_rank, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&south_rank, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&south_first, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD, &status);
    debug_printf("From %d (recovery) - received recovery data: north: %d, south: %d, south first: %d\n",
                 rank, north_rank, south_rank, south_first);

    load_worker_checkpoint(dead_rank);

    debug_printf("From %d (recovery) - successful loading, ack to master\n", rank);
    int tmp;
    MPI_Send(&tmp, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD);

    debug_printf("From %d (recovery) - entering worker loop\n", rank);
    worker_state();
}

void worker_entry() {
    /*
     * Задача рабочего процесса - проводить вычисления на своей части матрицы
     * При этом в конце каждой итерации требуется синхронизация с соседними процессами
     * На этапе синхронизации может обнаружиться, что один из соседей упал
     * В таком случае рабочий процесс отправляет процессу-мастеру запрос на восстановление
     * После восстановления рабочий процесс запоминает ранг восстановленного соседа
     */

    worker_init();
    save_worker_checkpoint();
    debug_printf("From %d (worker) - saved first checkpoint\n", rank);
    worker_state();
}


/****************************
 * WORKER PROCESS FUNCTIONS *
 ****************************/

void worker_init() {
    /*
     * Здесь происходит инициализация данных рабочего процесса
     */
    north_rank = worker_north_rank[rank - 1];
    south_rank = worker_south_rank[rank - 1];
    south_first = worker_south_first[rank - 1];

    free_workers_info();

    int num_rows = (N - 2) / worker_proc_num;
    start_row = num_rows * (rank - 1) + 1;
    last_row = start_row + num_rows - 1;
    last_row += rank == worker_proc_num ? (N - 2) % worker_proc_num : 0;

    it_num = 0;
    state = PENDING_COMPUTE;

    matrix_init();
    debug_printf("From %d (worker) - initialized, start row: %d, last_row: %d\n", rank, start_row, last_row);
}

void worker_state() {
    /*
     * Эта функция реализует конечный автомат состояний рабочего процесса, которых всего три:
     * 1) вычисление
     * 2) синхронизация с первым соседом
     * 3) синхронизация со вторым соседом
     *
     * В любом из состояний допускается сбой, в процессе обработки которого
     * восстанавливающий процесс загрузит последнюю контрольную точку и продолжит
     * исполнение, войдя в эту функцию
     */
    for (; it_num < MAX_ITERATIONS; it_num++) {
        debug_printf("From %d (worker) - it_num: %d, state: %d\n", rank, it_num, state);
        switch (state) {
            case PENDING_COMPUTE:
                suicide
                compute();
                state = PENDING_FIRST_SYNC;
                save_worker_checkpoint();
                debug_printf("From %d (worker) - saved checkpoint\n", rank);

            case PENDING_FIRST_SYNC:
                suicide
                worker_sync(1);
                state = PENDING_SECOND_SYNC;
                save_worker_checkpoint();
                debug_printf("From %d (worker) - saved checkpoint\n", rank);

            case PENDING_SECOND_SYNC:
                suicide
                worker_sync(0);
                state = PENDING_COMPUTE;
                save_worker_checkpoint();
                debug_printf("From %d (worker) - saved checkpoint\n", rank);
        }
    }

    int tmp;
    debug_printf("From %d (worker) - ack to master\n", rank);
    MPI_Send(&tmp, 1, MPI_INT, 0, WORKER_FINISH_TAG, MPI_COMM_WORLD);

    debug_printf("From %d (worker) - sending data to master\n", rank);
    MPI_Send(&start_row, 1, MPI_INT, 0, WORKER_FINISH_TAG, MPI_COMM_WORLD);
    MPI_Send(&last_row, 1, MPI_INT, 0, WORKER_FINISH_TAG, MPI_COMM_WORLD);
    MPI_Send(&A[start_row][0][0], (last_row - start_row + 1) * N * N, MPI_DOUBLE, 0, WORKER_FINISH_TAG, MPI_COMM_WORLD);
    MPI_Send(&eps, 1, MPI_DOUBLE, 0, WORKER_FINISH_TAG, MPI_COMM_WORLD);
    debug_printf("From %d (worker) - finished\n", rank);
}

void worker_sync(int first_sync) {
    /*
     * Здесь происходит синхронизация соседних процессов - обмен данными из граничных слоев
     * двух областей матрицы, занятых этими процессами
     */
    while (1) {
        MPI_Status status;
        int dest = south_first == first_sync ? south_rank : north_rank;
        debug_printf("From %d (worker) - entering sync with %d\n", rank, dest);
        void *sendbuf = south_first == first_sync ? &A[last_row][0][0] : &A[start_row][0][0];
        void *recvbuf = south_first == first_sync ? &A[last_row + 1][0][0] : &A[start_row - 1][0][0];

        if (dest != NULL_RANK) {
            int err = MPI_Sendrecv(sendbuf, N * N, MPI_DOUBLE, dest, first_sync ? FIRST_SYNC_TAG : SECOND_SYNC_TAG,
                                   recvbuf, N * N, MPI_DOUBLE, dest, first_sync ? FIRST_SYNC_TAG : SECOND_SYNC_TAG, MPI_COMM_WORLD, &status);
            if (err) {
                printf("From %d (worker) - Process %d appears to be dead\n", rank, dest);
                worker_recovery(dest, south_first == first_sync);
                continue;
            }
            debug_printf("From %d (worker) - successful sync with %d\n", rank, dest);
        } 
        break;
    } 
}

void free_workers_info() {
    if (worker_rank)        free(worker_rank);
    if (worker_north_rank)  free(worker_north_rank);
    if (worker_south_rank)  free(worker_south_rank);
    if (worker_south_first) free(worker_south_first);

    worker_rank = NULL;
    worker_north_rank = NULL;
    worker_south_rank = NULL;
    worker_south_first = NULL;
}


/*****************************
 * WORKER RECOVERY FUNCTIONS *
 *****************************/

void save_worker_checkpoint() {
    debug_printf("From %d (worker) - saving checkpoint: it_num: %d, state: %d\n", rank, it_num, state);
    char path[100];
    snprintf(path, 100, "CP/control_point_%d.bin", rank);

    FILE *cp_file = fopen(path, "wb");

    if (cp_file == NULL) {
        printf("From %d (worker) - could not save checkpoint\n", rank);
        raise(SIGTERM);
        return;
    }

    fwrite(&start_row, sizeof(start_row), 1, cp_file);
    fwrite(&last_row, sizeof(last_row), 1, cp_file);
    fwrite(&it_num, sizeof(it_num), 1, cp_file);
    fwrite(&state, sizeof(state), 1, cp_file);
    fwrite(&eps, sizeof(eps), 1, cp_file);

    int start = north_rank == NULL_RANK ? start_row : start_row - 1;
    int end = south_rank == NULL_RANK ? last_row : last_row + 1;

    fwrite(&A[start][0][0], sizeof(double), (end - start + 1) * N * N, cp_file);

    fclose(cp_file);
    sync();
}

void load_worker_checkpoint(int dead) {
    debug_printf("From %d (recovery) - loading checkpoint of %d\n", rank, dead);
    char path[100];
    snprintf(path, 100, "CP/control_point_%d.bin", dead);

    sync();
    FILE *cp_file = fopen(path, "rb");

    if (cp_file == NULL) {
        printf("From %d (recovery) - could not load checkpoint\n", rank);
        raise(SIGTERM);
        return;
    }

    fread(&start_row, sizeof(start_row), 1, cp_file);
    fread(&last_row, sizeof(last_row), 1, cp_file);
    fread(&it_num, sizeof(it_num), 1, cp_file);
    fread(&state, sizeof(state), 1, cp_file);
    fread(&eps, sizeof(eps), 1, cp_file);

    int start = north_rank == NULL_RANK ? start_row : start_row - 1;
    int end = south_rank == NULL_RANK ? last_row : last_row + 1;

    fread(&A[start][0][0], sizeof(double), (end - start + 1) * N * N, cp_file);

    fclose(cp_file);
}

void worker_recovery(int dead, int south) {
    int err = MPI_Send(&dead, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD);
    if (err) {
        printf("From %d (worker) - failed to send recovery request\n", rank);
        MPI_Abort(MPI_COMM_WORLD, RECOVERY_FAILED);
        return;
    }
    MPI_Status status;
    MPI_Recv(south ? &south_rank : &north_rank, 1, MPI_INT, 0, RECOVERY_REQ_TAG, MPI_COMM_WORLD, &status);
}


/******************************
 * MATRIX OPERATION FUNCTIONS *
 ******************************/

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
    debug_printf("From %d (worker) - started relax\n", rank);
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

    debug_printf("From %d (worker) - started resid\n", rank);
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
    debug_printf("From %d (worker) - resid eps: %lf\n", rank, eps);

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

    FILE * res = fopen("result_ft.txt", "w");
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
