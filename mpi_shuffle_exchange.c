#include<stdio.h>
#include<string.h>
#include"mpi.h"
#include<math.h>

int routingFn(int j, int i){
        return j+pow(2,i);
}

int routingFn2(int j){
        int t;
        t = j>>2;
        j = (j<<1);
        j = (j%8)|t;
        return j;
}


void main(int argc, char* argv[]){
        int my_rank,p,source,dest,tag=0;
        int tag2=1;
        MPI_Status status;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        int n;
        int my_value;
        int my_step;
        int total_step = log2(p);
        int x=1;
        if(my_rank==0){
                my_step=0;
                int arr[] = {1,1,1,1,1,1,1,1};
                int mask[] = {0,1,0,1,0,1,0,1};
                int mask_n[] = {0,0,0,0,0,0,0,0};
                my_value=arr[0];
                int i;
                for(my_step=0;my_step<total_step;my_step++){
                        for(i=1;i<p;i++){
                                int t=arr[i];
                                int m=mask[i];
                                MPI_Send(&t, 1, MPI_INT,i,tag,MPI_COMM_WORLD);
                                MPI_Send(&m, 1, MPI_INT,i,tag+1,MPI_COMM_WORLD);
                        }

                        if(mask[my_rank]){
                                arr[0] *= x;
                        }
                        for(i=1;i<p;i++){
                                int y;
                                MPI_Recv(&y,1,MPI_INT,i,tag+2,MPI_COMM_WORLD,&status);
                                arr[i] = y;
                        }

                        for(i=0;i<p;i++){
                                int t=routingFn2(i);
                                mask_n[t]=mask[i];
                        }
                        for(i=0;i<p;i++){
                                mask[i]=mask_n[i];
                                printf("%d ",mask[i]);
                        }
                        printf("\n");
                }
                my_value = arr[0];

                for(my_step=0;my_step<total_step;my_step++){
                        for(i=1;i<p;i++){
                                int t = arr[i];
                                MPI_Send(&t, 1, MPI_INT,i,tag,MPI_COMM_WORLD);
                                MPI_Send(&my_step,1,MPI_INT,i,tag+1,MPI_COMM_WORLD);
                        }

                        int ns;
                        ns = routingFn(my_rank,my_step);
                        int k = arr[my_rank];
                        MPI_Send(&k, 1, MPI_INT,ns,tag+3,MPI_COMM_WORLD);
                        for(i=1;i<p;i++){
                                int y;
                                MPI_Recv(&y,1,MPI_INT,i,tag+2,MPI_COMM_WORLD,&status);

                                arr[i] = y;
                        }
                }
                for(i=0;i<p-1;i++){
                        printf("Result for P[%d] : %d\n",i,arr[i]);
                }
        }
        else{
                int j;
                for(j=0;j<total_step;j++){
                        source = 0;
                        MPI_Recv(&my_value,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                        MPI_Recv(&my_step,1,MPI_INT,source,tag+1,MPI_COMM_WORLD,&status);
                        if(my_step){
                                my_value*=x;
                        }
                        MPI_Send(&my_value,1,MPI_INT,source,tag+2,MPI_COMM_WORLD);
                        x*=x;
                }

                for(j=0;j<total_step;j++){
                        source = 0;
                        MPI_Recv(&my_value,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                        MPI_Recv(&my_step,1,MPI_INT,source,tag+1,MPI_COMM_WORLD,&status);
                        int ns;
                        if(my_rank%2==0){
                                ns = routingFn(my_rank,my_step);
                                if(ns<8){
                                        MPI_Send(&my_value, 1, MPI_INT,ns,tag+3,MPI_COMM_WORLD);
                                }
                                int incoming;
                                int incomingP = my_rank - pow(2,my_step);
                                if(my_rank>=pow(2,my_step)){
                                        MPI_Recv(&incoming, 1, MPI_INT,incomingP,tag+3,MPI_COMM_WORLD,&status);
                                }

                                my_value += incoming;

                                MPI_Send(&my_value, 1, MPI_INT,source,tag+2,MPI_COMM_WORLD);
                        }
                        else{
                                ns = routingFn(my_rank,my_step);
                                int incoming;
                                int incomingP = my_rank - pow(2,my_step);
                                if(my_rank>=pow(2,my_step)){
                                        MPI_Recv(&incoming, 1, MPI_INT,incomingP,tag+3,MPI_COMM_WORLD,&status);
                                }
                                if(ns<8){
                                        MPI_Send(&my_value, 1, MPI_INT,ns,tag+3,MPI_COMM_WORLD);
                                }
                                my_value += incoming;

                                MPI_Send(&my_value, 1, MPI_INT,source,tag+2,MPI_COMM_WORLD);
                        }
                }

        }

        MPI_Finalize();

}

