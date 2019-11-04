#include<stdio.h>
#include<string.h>
#include<math.h>
#include"mpi.h"
void main(int argc,char*argv[])
{
        int my_rank,p,source,dest,tag1=1,tag2=2,tag3=3,tag4=4,tag5=5,tag6=6,tag7=7,tag8=8;
        MPI_Status status;
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        MPI_Comm_size(MPI_COMM_WORLD,&p);
        int a[100];
        int n;
        if(my_rank==0)
        {
                printf("enter number of elemenets: \n");
                scanf("%d",&n);
                int i;
                printf("enter elements: ");
                for(i=0;i<n;i++)
                a[i]=i+1;
                int j=2;
                for(i=0;j<=n;)
                {
                        int f;
                        for(dest=1;dest<p;dest++)
                        {
                                int y=1<<i;
                                int src=dest-y;
                                MPI_Send(&src,1,MPI_INT,dest,tag6,MPI_COMM_WORLD);

                        }
                        int x=1<<i;
                        dest=0+x;
                        MPI_Send(&i,1,MPI_INT,dest,tag1,MPI_COMM_WORLD);
                        MPI_Send(a,100,MPI_INT,dest,tag4,MPI_COMM_WORLD);
                        MPI_Recv(a,100,MPI_INT,dest,tag3,MPI_COMM_WORLD,&status);
                        i++;
                        j=1<<(i+1);
                }
                printf("prefix sum in process %d is %d\n",my_rank,a[0]);
        }
        else
        {
                source=0;
                int src;
                MPI_Recv(&src,1,MPI_INT,source,tag6,MPI_COMM_WORLD,&status);
                if(src>=0)
                if(src>=0)
                {
                source=src;
                int i;
                MPI_Recv(&i,1,MPI_INT,source,tag1,MPI_COMM_WORLD,&status);
                MPI_Recv(a,100,MPI_INT,source,tag4,MPI_COMM_WORLD,&status);
                int x=1<<i;
                dest=my_rank+x;
                if(dest<p)
                {
                        MPI_Send(&i,1,MPI_INT,dest,tag2,MPI_COMM_WORLD);
                        MPI_Send(a,100,MPI_INT,dest,tag5,MPI_COMM_WORLD);
                        MPI_Recv(a,100,MPI_INT,dest,tag7,MPI_COMM_WORLD,&status);
                }
                a[my_rank]+=a[source];
                int j=1<<(i+1);
                if(j==p)
                {
                        printf("prefix sum in process %d is %d\n",my_rank,a[my_rank]);
                }
                MPI_Send(a,100,MPI_INT,source,tag3,MPI_COMM_WORLD);
                }
        }
        MPI_Finalize();
}
