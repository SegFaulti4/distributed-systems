#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define N_ROWS 8
#define N_COLS 8
#define NULL_RANK (-1)

#define Min(a,b) ((a)<(b)?(a):(b))
#define Max(a,b) ((a)>(b)?(a):(b))

int size = N_ROWS * N_COLS;
int rank;

void decide_max(int *data, int *best_rank, int *other_data, int *other_rank) {
    if (*other_data > *data) {
        *data = *other_data;
        *best_rank = *other_rank;
    }
}

void communicate(int other0, int other1, int *recv_data, int *recv_rank, int *data, int *best_rank, int send, MPI_Comm comm) {
    int other[2];
    other[0] = other0;
    other[1] = other1;
    int other_rank;
    MPI_Cart_rank(comm, other, &other_rank);

    if (send) {
        MPI_Send(best_rank, 1, MPI_INT, other_rank, 0, comm);
        MPI_Send(data, 1, MPI_INT, other_rank, 0, comm);
    } else {
        MPI_Recv(recv_rank, 1, MPI_INT, other_rank, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(recv_data, 1, MPI_INT, other_rank, 0, comm, MPI_STATUS_IGNORE);
        decide_max(data, best_rank, recv_data, recv_rank);
    }
}

void reduction(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int send = 1;
    int other[2];
    int other_rank;
    int other_data;
    if (coord[0] >= 0 && coord[0] <= 3) {
        if (coord[0] > 0) {
            communicate(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
        }
        communicate(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
    } else if (coord[0] <= 7 && coord[0] >= 5) {
        if (coord[0] < 7) {
            communicate(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
        }
        communicate(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
    } else {
        communicate(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
        communicate(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);

        if (coord[1] >= 0 && coord[1] <= 3) {
            if (coord[1] > 0) {
                communicate(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, !send, comm);
            }
            communicate(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, send, comm);
        } else if (coord[1] <= 7 && coord[1] >= 5) {
            if (coord[1] < 7) {
                communicate(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, !send, comm);
            }
            communicate(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, send, comm);
        } else {
            // coord[0] = 4, coord[1] = 4;
            communicate(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, !send, comm);
            communicate(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, !send, comm);
        }
    }
}

void broadcast(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int send = 1;
    int other[2];
    int other_rank;
    int other_data;
    if (coord[0] >= 0 && coord[0] <= 3) {
        communicate(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
        if (coord[0] > 0) {
            communicate(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
        }
    } else if (coord[0] <= 7 && coord[0] >= 5) {
        communicate(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
        if (coord[0] < 7) {
            communicate(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
        }
    } else {
        if (coord[1] >= 0 && coord[1] <= 3) {
            communicate(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, !send, comm);
            if (coord[1] > 0) {
                communicate(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, send, comm);
            }
        } else if (coord[1] <= 7 && coord[1] >= 5) {
            communicate(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, !send, comm);
            if (coord[1] < 7) {
                communicate(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, send, comm);
            }
        } else {
            // coord[0] = 4, coord[1] = 4;
            communicate(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, send, comm);
            communicate(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, send, comm);
        }

        communicate(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
        communicate(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    int dims[2] = {N_ROWS, N_COLS};
    int periods[2] = {0};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
    MPI_Cart_coords(comm, rank, 2, coords);

    srand(rank);
    int data = rand() % 1000000;
    int best_rank = rank;

    if (rank == 0) { printf("Generated data:\n"); }
    MPI_Barrier(comm);

    for (int i = 0; i < size; i++) {
        if (i == rank) {
            printf("rank: %d \tcoords: %d, %d\tdata: %d\n", rank, coords[0], coords[1], data);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    reduction(&data, &best_rank, comm, coords);
    MPI_Barrier(comm);
    if (coords[0] == 4 && coords[1] == 4) { printf("\nMax data value: %d\nMax data rank: %d\n", data, best_rank); }

    broadcast(&data, &best_rank, comm, coords);
    int best_coords[2];
    MPI_Cart_coords(comm, best_rank, 2, best_coords);
    if (rank == 0) { printf("\nData after broadcast:\n"); }
    MPI_Barrier(comm);

    for (int i = 0; i < size; i++) {
        if (i == rank) {
            printf("rank: %d \tbest rank: %d\tbest coords: %d, %d\tdata: %d\n",
                   rank, best_rank, best_coords[0], best_coords[1], data);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;
}
