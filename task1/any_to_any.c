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

// https://stackoverflow.com/a/15327567
int ceil_log2(unsigned long long x) {
    static const unsigned long long t[6] = {
        0xFFFFFFFF00000000ull,
        0x00000000FFFF0000ull,
        0x000000000000FF00ull,
        0x00000000000000F0ull,
        0x000000000000000Cull,
        0x0000000000000002ull
    };

    int y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 32;
    int i;

    for (i = 0; i < 6; i++) {
        int k = (((x & t[i]) == 0) ? 0 : j);
        y += k;
        x >>= k;
        j >>= 1;
    }

    return y;
}

void init_binomial_heap(int *parent, int *children_num, int **children, int null_parent) {
    for (int i = 0; i < size; i++) {
        parent[i] = null_parent;
    }

    int *queue = calloc(size, sizeof(*queue));
    int *next_queue = calloc(size, sizeof(*next_queue));
    int depth = ceil_log2(size);
    children_num[0] = depth;
    children[0] = malloc(sizeof(**children) * children_num[0]);
    queue[0] = children_num[0];

    int next = 1;
    int reserved = children_num[0];
    int step = 1;

#ifdef DEBUG
    if (rank == 0) { printf("Sequence of messages during broadcast:\n"); }
#endif
    while (reserved) {
        memcpy(next_queue, queue, size * sizeof(*queue));
#ifdef DEBUG
        if (rank == 0) { printf("step %d: ", step); }
#endif
        for (int i = 0; i < size; i++) {
            if (queue[i] > 0) {
                children[i][children_num[i] - next_queue[i]] = next;
                parent[next] = i;
                next_queue[i]--;
#ifdef DEBUG
                if (rank == 0) { printf("%d -> %d ", i, next); }
#endif
                int child_count = Max(Min(size - next - reserved, next_queue[i]), 0);
                if (child_count == 0) {
                    children[next] = NULL;
                } else {
                    children[next] = malloc(sizeof(**children) * child_count);
                    next_queue[next] = child_count;
                    reserved += child_count;
                }
                children_num[next] = child_count;
                next++;
                reserved--;
            }
        }
#ifdef DEBUG
        if (rank == 0) { printf("\n"); }
#endif
        memcpy(queue, next_queue, size * sizeof(*queue));
        step++;
    }

    free(next_queue);
    free(queue);
}

void reduce_with_max(int *data, int *best_rank, MPI_Comm comm, int parent, int children_num, int *children) {
    if (children_num != 0) {
        int tmp;
        int other_rank;
        for (int i = 0; i < children_num; i++) {
            MPI_Recv(&tmp, 1, MPI_INT, children[i], 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&other_rank, 1, MPI_INT, children[i], 0, comm, MPI_STATUS_IGNORE);
            if (tmp > *data) {
                *data = tmp;
                *best_rank = other_rank;
            }
        }
    }
    if (parent != NULL_RANK) {
        MPI_Send(data, 1, MPI_INT, parent, 0, comm);
        MPI_Send(best_rank, 1, MPI_INT, parent, 0, comm);
    }
}

void broadcast_data(int *data, int *best_rank, MPI_Comm comm, int parent, int children_num, int *children) {
    if (parent != NULL_RANK) {
        MPI_Recv(data, 1, MPI_INT, parent, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(best_rank, 1, MPI_INT, parent, 0, comm, MPI_STATUS_IGNORE);
    }
    if (children_num != 0) {
        for (int i = 0; i < children_num; i++) {
            MPI_Send(data, 1, MPI_INT, children[i], 0, comm);
            MPI_Send(best_rank, 1, MPI_INT, children[i], 0, comm);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *parent_rank = malloc(sizeof(*parent_rank) * size);
    int *children_num_rank = malloc(sizeof(*children_num_rank) * size);
    int **children_rank = malloc(sizeof(*children_rank) * size);
    init_binomial_heap(parent_rank, children_num_rank, children_rank, NULL_RANK);

    MPI_Comm comm;
    int dims[2] = {N_ROWS, N_COLS};
    int periods[2] = {0};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
    MPI_Cart_coords(comm, rank, 2, coords);

    int parent = parent_rank[rank];
    int children_num = children_num_rank[rank];
    int *children = children_rank[rank];

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

    reduce_with_max(&data, &best_rank, comm, parent, children_num, children);
    MPI_Barrier(comm);
    if (rank == 0) { printf("\nMax data value: %d\nMax data rank: %d\n", data, best_rank); }

    broadcast_data(&data, &best_rank, comm, parent, children_num, children);
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

    for (int i = 0; i < size; i++) {
        if (children_num_rank[i] != 0) {
            free(children_rank[i]);
        }
    }
    free(children_rank);
    free(children_num_rank);
    free(parent_rank);

    MPI_Finalize();
    return 0;
}
