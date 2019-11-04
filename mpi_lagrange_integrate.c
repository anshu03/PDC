#include<stdio.h>
#include<string.h>
#include"mpi.h"
void main(int argc,char*argv[])
{
        int my_rank,p,source,dest,tag1=1,tag2=2,tag3=3,tag4=4,tag5=5,tag6=6,tag7=7;
        MPI_Status status;
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        MPI_Comm_size(MPI_COMM_WORLD,&p);
        float a[100];
        float b[100];
        int n;
        if(my_rank==0)
        {
                printf("enter number of elemenets: \n");
                scanf("%d",&n);
                int i;
                for(i=0;i<n;i++)
                {
                        a[i]=i+1;
                        b[i]=(n/2)-i;
                        printf("%f,%f\n",a[i],b[i]);
                }
                float x;
                printf("enter value of x:\n");
                scanf("%f",&x);
                int c=n/p;
                int rem=n-(c*(p-1));
                float msum=0.0;
                int j;
                for(i=0;i<rem;i++)
                {
                        float z=1.0;
                        for(j=0;j<n;j++)
                        {
                                if(j!=i)
                                {
                                        z=(float)z*((x-a[j])/(a[i]-a[j]));
                                }
                        }
                        msum=msum+z*b[i];
                }
                for(dest=1;dest<p;dest++)
                {
                        i=rem;
                        rem=rem+c;
                        MPI_Send(&i,1,MPI_INT,dest,tag1,MPI_COMM_WORLD);
                        MPI_Send(&c,1,MPI_INT,dest,tag2,MPI_COMM_WORLD);
                        MPI_Send(&x,1,MPI_FLOAT,dest,tag5,MPI_COMM_WORLD);
                        MPI_Send(a,100,MPI_FLOAT,dest,tag4,MPI_COMM_WORLD);
                        MPI_Send(b,100,MPI_FLOAT,dest,tag6,MPI_COMM_WORLD);
                        MPI_Send(&n,1,MPI_INT,dest,tag7,MPI_COMM_WORLD);
                        float y;
                        MPI_Recv(&y,1,MPI_FLOAT,dest,tag3,MPI_COMM_WORLD,&status);
                        msum+=y;
                }
                printf("function value is %f\n",msum);
        }
        else
        {
                source=0;
                int i,c,n;
                float x;
                MPI_Recv(&i,1,MPI_INT,source,tag1,MPI_COMM_WORLD,&status);
                MPI_Recv(&c,1,MPI_INT,source,tag2,MPI_COMM_WORLD,&status);
                MPI_Recv(&x,1,MPI_FLOAT,source,tag5,MPI_COMM_WORLD,&status);
                MPI_Recv(a,100,MPI_FLOAT,source,tag4,MPI_COMM_WORLD,&status);
                MPI_Recv(b,100,MPI_FLOAT,source,tag6,MPI_COMM_WORLD,&status);
                MPI_Recv(&n,1,MPI_INT,source,tag7,MPI_COMM_WORLD,&status);
                int j,k;
                float ssum=0.0;
                for(k=i;k<i+c;k++)
                {
                        float z=1.0;
                        for(j=0;j<n;j++)
                        {
                                if(k!=j)
                                {
                                        z=(float)z*((x-a[j])/(a[k]-a[j]));
                                }
                        }
                        ssum+=z*b[k];
                }
                MPI_Send(&ssum,1,MPI_FLOAT,source,tag3,MPI_COMM_WORLD);

        }
        MPI_Finalize();
}

