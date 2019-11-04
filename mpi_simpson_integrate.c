#include  <stdio.h>
#include  <math.h>
#include  <mpi.h>

#define approx_val  2.19328059
#define N 32

double integrate_f(double);
double simpson(int, double, double, double);

int main(int argc, char *argv[]) {
  int Procs;
  int my_rank;
  double total; 
  double exact_val_of_Pi, pi, y, processor_output_share[8], x1, x2, l, sum;
  int i;
  MPI_Status status;
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  MPI_Comm_size(MPI_COMM_WORLD, &Procs);
  
  x1 = ((double) my_rank)/((double) Procs);
  x2 = ((double) (my_rank + 1))/((double) Procs);
  
  l = 1.0/((double) (2 * N * Procs));
  sum = 0.0;
  for(i = 1; i < N ; i++)
  {
    y = x1 + (x2 - x1)*((double) i)/((double) N);
    
    sum = (double) simpson(i, y, l, sum);
  }

  sum += (integrate_f(x1) + integrate_f(x2))/2.0;
  total = sum;
  
  if(my_rank == 0)
  {
    processor_output_share[0] = total;
    
    for(i = 1; i < Procs; i++)
      MPI_Recv(&(processor_output_share[i]), 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
  }
  else
  {
    MPI_Send(&total, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  if(my_rank == 0)
  {
    for(i = 0; i < Procs; i++)
      pi += processor_output_share[i];
    pi *= 2.0 * l/ 3.0;
    printf("-------------------------------------------------\n");
    printf("The computed Pi of the integral for %d grid points is  %25.16e \n",
           (N * Procs), pi);
 
#if 1
      exact_val_of_Pi = 4.0 * atan(1.0);
#endif

#if 0
      exact_val_of_Pi = 4.0 * log(approx_val); 
#endif
      printf("The error or the discrepancy between exact and computed value of Pi : %25.16e\n",
             fabs(pi - exact_val_of_Pi));
      printf("-------------------------------------------------\n");
  }
  
  MPI_Finalize();
}
 

double integrate_f(double x) {
  return 4.0/(1.0 + x * x);     
}

double simpson(int i, double y, double l, double sum) {
  sum += integrate_f(y);
  sum += 2.0 * integrate_f(y - l);
  if(i == (N - 1))
    sum += 2.0 * integrate_f(y + l);
  return sum;
}


