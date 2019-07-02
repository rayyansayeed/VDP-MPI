#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>


/* p = vanderpol(alpha,sigma,M,N)
simulate Van der Pol oscillator with noise
alpha = location of sinks at +/-alpha,0
sigma = strength of noise
M = number of time steps
N = number of trials */

// Set parameter values, initialize

double alpha;
double T = 10.0;
int N, M;
double sigma;
double dt;
double M_PI = 3.141592;

// initialize vdp function

double vdp(double, double, double, double);

int main(int argc, char *argv[]){

    // MPI initialization

	MPI_Init(&argc, &argv);
	int MPI_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
	int MPI_size;
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

	// Set up runtime clock, start

	clock_t start_time = clock();

	// Prepare args to be taken as inputs

	N = atoi(argv[1]);
	M = atoi(argv[2]);
	sigma = atof(argv[3]);
	alpha = 1;
	dt = T / M;

	// Initialize random seed

	long int seed = (long int)time(NULL);
	if (argc >= 5){
		seed = atol(argv[4]);
	}
	long int seed_MPI;

	// Allocate memory for dist p and count

	double *p = (double *)malloc((M+1) * sizeof(double));

	// Allocate memory for count in local and global settings

	int *p_count_local = (int *)malloc((M+1) * sizeof(int));
	int *p_count_global = (int *)malloc((M+1) * sizeof(int));

	// Zero counts to start

	for (int i = 0; i < M + 1; ++i)
	{
		p_count_local[i] = 0;
		p_count_global[i] = 0;
	}

	// Begin Euler-Maruyama scheme

	if (MPI_rank == 0) {

        // Random seeding set up

		seed_MPI = seed + MPI_rank;
		srand(seed_MPI);

		// Address remaining samples

		int rem_samples = N - (N / (MPI_size-1))*(MPI_size - 1);

        // No remaining samples, set local count to 0

		if (rem_samples == 0) {

			for (int i = 0; i < M+1; ++i)
				p_count_local[i] = 0;
		}

		// Remaining samples is nonzero, simulate through

		else {

			for (int i = 0; i < rem_samples; ++i){

				// Generate Normal dist and initial pts

				double R1_X = (double)rand() / RAND_MAX;
				double R2_X = (double)rand() / RAND_MAX;

				double X = 0.1 * sqrt(-2 * log(R1_X)) * cos(2 * M_PI*R2_X);
				double Y = 0.0;

				for (int j = 0; j < M + 1; ++j){

					// d1y = ((alpha^2 - x^2)x - y)dt

					double d1Y = (((alpha*alpha) - X * X) * (X) - Y) * dt;

					// dW ~ sqrt(dt) * N(0,1)

					double R1 = (double)rand() / RAND_MAX;
					double R2 = (double)rand() / RAND_MAX;
					double dW = sqrt(dt) * sqrt(-2 * log(R1)) * cos(2 * M_PI*R2);

					// d2y = sigma * x * dW

					double d2Y = sigma * X * dW;

					// Increment through, update local count

					X += Y * dt;
					Y += (d1Y + d2Y);

					if (vdp(X, Y, -alpha, 0) <= 0.5 || vdp(X, Y, alpha, 0) <= 0.5)
						++p_count_local[j];
				}

			}
		}
	}
	else {
		seed_MPI = seed + MPI_rank;
		srand(seed_MPI);

		// Equally distributed sample size for remaining samples

		for (int i = 0; i < N/MPI_size; ++i){

			// generate Normal dist and initial pts

			double R1_X = (double)rand() / RAND_MAX;
			double R2_X = (double)rand() / RAND_MAX;
			double X = 0.1 * sqrt(-2 * log(R1_X)) * cos(2 * M_PI*R2_X);
			double Y = 0.0;

			for (int j = 0; j < M + 1; ++j)
			{
				// d1y = ((alpha^2 - x^2)x - y)dt

				double d1Y = ((alpha*alpha - X * X) * X - Y) * dt;

				// dW ~ sqrt(dt) * N(0,1)

				double R1 = (double)rand() / RAND_MAX;
				double R2 = (double)rand() / RAND_MAX;
				double dW = sqrt(dt) * sqrt(-2 * log(R1)) * cos(2 * M_PI*R2);
				double d2Y = sigma * X * dW;

				// Increment through, update local count

				X += Y * dt;
				Y += (d1Y + d2Y);
				if (vdp(X, Y, -alpha, 0) <= 0.5 || vdp(X, Y, alpha, 0) <= 0.5)
					++p_count_local[j];
			}

		}


	}

	// Gather stored result
	MPI_Reduce(p_count_local, p_count_global, M + 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	// Output result

	double *p_out = (double *)malloc((101) * sizeof(double));

	// Create and open target file

	FILE *fileid = fopen("VDP.out", "w");

	for (int j = 0; j <= 100; ++j){
		p_out[j] = p[(int)((double)j*M / 100)];
	}

	// Write to target file

	fwrite(p_out, sizeof(double), 101, fileid);

	// End runtime clock, output runtime to console
	clock_t end_time = clock();
	if (MPI_rank == 0)
		printf("Runtime: %1.4fs\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);


	// Free all

	free(p_count_local);
	free(p_count_global);
	free(p);
	free(p_out);
	MPI_Finalize();
	return 0;
}

double vdp(double X1, double Y1, double X2, double Y2)
{
	return sqrt((X1 - X2)*(X1 - X2) + (Y1 - Y2)*(Y1 - Y2));
}
