#include<stdio.h>
#include<string.h>
#include<math.h>
#include "mpi.h"

void main(int argc, char* argv[] )
{
        int my_rank,p,a[100][100],b[100][100],source,dest;
        int tag=0;
        MPI_Status status;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        if(my_rank!=0)
        {
                int x,y,z,x1,y1,z1;
                MPI_Recv(&x,1,MPI_INT,0,tag, MPI_COMM_WORLD, &status);
                MPI_Recv(&y,1,MPI_INT,0,tag, MPI_COMM_WORLD, &status);
                MPI_Recv(&z,1,MPI_INT,0,tag, MPI_COMM_WORLD, &status);
                int n=sqrt(p-1);
                int k;
                int i=(my_rank/n)+1;
                if(my_rank%n==0)
                i--;
                int j=(my_rank%n);
                if(my_rank%n==0)
                j=n;
                for(k=1;k<=n-1;k++)
                {

                        if(i>k)
                        {
                                int dest=my_rank-1;
                                if(dest%n==0)
                                dest=dest+n;
                                int source=my_rank+1;
                                if(source%n==1)
                                source=source-n;
                                MPI_Send(&x,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
                                MPI_Recv(&x,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                        }
                        if(j>k)
                        {
                                int dest=my_rank-n;
                                if(dest<=0)
                                dest=dest+n*n;
                                int source=my_rank+n;
                                if(source>n*n)
                                source=source-n*n;
                                MPI_Send(&y,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
                                MPI_Recv(&y,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                        }
                }
                for(k=1;k<=n;k++)
                {
                        z=z+x*y;
                        int dest=my_rank-1;
                        if(dest%n==0)
                        dest=dest+n;
                        int source=my_rank+1;
                        if(source%n==1)
                        source=source-n;
                        MPI_Send(&x,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
                        MPI_Recv(&x,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);

                        dest=my_rank-n;
                        if(dest<=0)
                        dest=dest+n*n;
                        source=my_rank+n;
                        if(source>n*n)
                        source=source-n*n;
                        MPI_Send(&y,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
                        MPI_Recv(&y,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);

                }
                MPI_Send(&z,1,MPI_INT,0,tag,MPI_COMM_WORLD);
        }
        else
        {
                int n;
                printf("Enter dimension of matrix: \n");
                scanf("%d",&n);
                int i,j;
                int count=0;
                for(i=0;i<n;i++)
                {
                        for(j=0;j<n;j++)
                        {
                                count++;
                                if(j==i)
                                a[i][j]=1;
                                else
                                a[i][j]=0;

                                b[i][j]=2;
                                int x=a[i][j];
                                int y=b[i][j];
                                int z=0;
                                MPI_Send(&x,1,MPI_INT,count,tag,MPI_COMM_WORLD);
                                MPI_Send(&y,1,MPI_INT,count,tag,MPI_COMM_WORLD);
                                MPI_Send(&z,1,MPI_INT,count,tag,MPI_COMM_WORLD);
                        }
                }
                int c[100][100];
                for(i=1;i<=n;i++)
                {
                        for(j=1;j<=n;j++)
                        {
                                int source=(i-1)*n+j;
                                int x;
                                MPI_Recv(&x,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                                c[i-1][j-1]=x;
                                printf("%d ",x);
                        }
                        printf("\n");
                }
        }
        MPI_Finalize();
}

