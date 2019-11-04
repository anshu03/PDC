#include<stdio.h>
#include<string.h>
#include "mpi.h"

void main(int argc, char* argv[] )
{
        int my_rank,p,a[100],source,dest;
        int tag=0;
        MPI_Status status;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        if(my_rank!=0)
        {
                int x,z;
                MPI_Recv(&x,1,MPI_INT,0,tag, MPI_COMM_WORLD, &status);
                int j;
                int k=(p-1)/2;
                if((p-1)%2!=0)
                k++;
                for(j=1;j<=k;j++)
                {
                        if(my_rank+1<p && my_rank%2==1)
                        {
                                MPI_Send(&x,1,MPI_INT,my_rank+1,tag,MPI_COMM_WORLD);
                                MPI_Recv(&x,1,MPI_INT,my_rank+1,tag,MPI_COMM_WORLD,&status);
                        }
                        if(my_rank-1>=1 && my_rank%2==0)
                        {
                                MPI_Recv(&z,1,MPI_INT,my_rank-1,tag,MPI_COMM_WORLD,&status);
                                if(z>x)
                                {
                                        int temp=z;
                                        z=x;
                                        x=temp;
                                }
                                MPI_Send(&z,1,MPI_INT,my_rank-1,tag,MPI_COMM_WORLD);
                        }
                        if(my_rank+1<p && my_rank%2==0)
                        {
                                MPI_Send(&x,1,MPI_INT,my_rank+1,tag,MPI_COMM_WORLD);
                                MPI_Recv(&x,1,MPI_INT,my_rank+1,tag,MPI_COMM_WORLD,&status);
                        }
                        if(my_rank-1>=1 && my_rank%2==1)
                        {
                                MPI_Recv(&z,1,MPI_INT,my_rank-1,tag,MPI_COMM_WORLD,&status);
                                if(z>x)
                                {
                                        int temp=z;
                                        z=x;
                                        x=temp;

                                }
                                MPI_Send(&z,1,MPI_INT,my_rank-1,tag,MPI_COMM_WORLD);
                        }
                }
                MPI_Send(&x,1,MPI_INT,0,tag,MPI_COMM_WORLD);
        }
        else
        {
                int n;
                printf("Enter total elements: \n");
                scanf("%d",&n);
                int i;
                for(i=0;i<n;i++)
                        scanf("%d",&a[i]);
                for(i=0;i<n;i++)
                {
                        int b=a[i];
                        MPI_Send(&b,1,MPI_INT,i+1,tag,MPI_COMM_WORLD);
                }
                for(source=1;source<p;source++)
                {
                        int x;
                        MPI_Recv(&x,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                        a[source-1]=x;
                        printf("Message from process %d element = %d\n",source-1,x);
                }
        }
        MPI_Finalize();
}

