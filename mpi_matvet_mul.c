#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

void multi(int Count,float *Sum,float Vec[],float Data[],int Column)
{
  int i=0,j=0,k=0;
  while(i<Count)
  {
    Sum[i]=0;
    for(j=0;j<Column;j++)
    {
       Sum[i] = Sum[i] + Data[k] * Vec[j];
       k++;
    }
    i++;
  }
}

int main(int argc,char *argv[])
{
  int rank,size,*sendcount,*displace,*reccount;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size); 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Status status;
  FILE *fp;
  char c;
  int i,j,k=0,count=0,row=0,column=0;
  float n=0,*sum,*rec_data,*data,*vec;
  sendcount = (int*)calloc(sizeof(int),size);
  reccount  = (int*)calloc(sizeof(int),size);
  displace  = (int*)calloc(sizeof(int),size);
  if(rank==0)
  {
    fp=fopen("matrix.txt","r");
    while(fscanf(fp,"%f",&n)!=-1)
    { 
      c=fgetc(fp);
      if(c=='\n'){ row=row+1; }
      count++;
    }
    column=count/row; 
    printf("Row=%d column=%d proc=%d\n",row,column,size);
    float mat[row][column];
    fseek( fp, 0, SEEK_SET );
    data = (float*)calloc(sizeof(float),row*column);
    vec  = (float*)calloc(sizeof(float),column);
    for(i=0;i<row;i++) 
    {
       for(j=0;j<column;j++)
       {
          fscanf(fp,"%f",&mat[i][j]); 
          data[k] = mat[i][j];
          k++;
       }
    }
    fclose(fp);
    fp    =  fopen("vector.txt","r"); 
    count = 0;
    while(fscanf(fp,"%f",&n)!=-1){  count++; }
    printf("length of vector = %d\n",count);
    if(column!=count) { printf("Dimensions do not match.\nCode Terminated"); MPI_Abort(MPI_COMM_WORLD,0); }
    fseek( fp, 0, SEEK_SET );
    for(i=0;i<column;i++)
    {
      fscanf(fp,"%f",&vec[i]); 
    }
    fclose(fp);
    count=0;
    while(1)
    {
      for(i=0;i<size;i++)
      {
        sendcount[i] = sendcount[i]+1;
        count++;
        if(count==row) break;  
      }
      if(count==row)  break; 
    }
    for(i=1;i<size;i++)
    {
      displace[i] = displace[i-1] + sendcount[i-1]*column;
      sendcount[i-1] = sendcount[i-1] * column;
    }
    sendcount[size-1] = sendcount[size-1] * column;
    for(i=0;i<size;i++)
      printf("sendcout=%d disp=%d\n",sendcount[i],displace[i]);
  }
  MPI_Bcast(&row,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&column,1,MPI_INT,0,MPI_COMM_WORLD);
  if(rank!=0) 
  {
    vec = (float *)malloc(sizeof(float) * column);
  }
  MPI_Bcast(vec,column,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(sendcount,size,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(displace,size,MPI_INT,0,MPI_COMM_WORLD);  
  rec_data=(float*)calloc(sizeof(float),sendcount[rank]);
  MPI_Scatterv(data,sendcount,displace,MPI_FLOAT,rec_data,sendcount[rank],MPI_FLOAT,0,MPI_COMM_WORLD);
  count=sendcount[rank]/column;
  sum=(float*)calloc(sizeof(float),count);
  multi(count,sum,vec,rec_data,column);
  float *result=(float *)calloc(sizeof(float),row);
  int disp[size];
  disp[0]=0;
  reccount[0]=sendcount[0]/column;  
  for(i=1;i<size;i++)
  {
    disp[i] = disp[i-1] + sendcount[i-1]/column; 
    reccount[i]=sendcount[i]/column;
  }
  MPI_Gatherv(sum,count,MPI_FLOAT,result,reccount,disp,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(rank==0)
  {
    printf("\nMatrix Vector Multiplication is:\n");
    for(i=0;i<row;i++)
    {
       printf("%.3f\n",result[i]);
    }     
  }
  free(vec);
  free(sum);
  free(sendcount);
  free(displace);
  free(reccount);
  free(rec_data); 
  MPI_Finalize();
  return 0;
}
