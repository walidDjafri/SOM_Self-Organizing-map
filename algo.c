#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//*** la Structure de données pour ma configuration de base ***

 struct Config_base
 {
   int v_size; //*** La variable de la taille du vecteur ***
   int nbr_n_c; //*** Le nombre de neuron sur colone ***
   int nbr_n_l; //*** Le nombre de neuron map sur ligne ***
   int T_neuron; //*** Le total neuron (nbr_n_l * nbr_n_c) taille de notre map ***
   int nb_iteration; //*** le nombre d'iterations qu'on calcule avec la methode donnée ***
   double minAlpha;//*** The Starting value alpha  ***
   int train; //*** Le nombre d'opération lors de l'entrainement ***
   int ftrain; //*** Le numero de l'operation dans la 1ere couche ***
 }conf_B;

//*** La structure d'un neuron ***
 
struct node
{
  double act; // *** The euclidian distance ***
  char *etiq;
  double *w;//*** le weight des connections synaptique ***
};

typedef struct node t_node;

//*** le BMU (Best-Matching Unit) ***

struct bmu {
	double act; //*** euclidian distance ***
	int r;
	int c;
};

typedef struct bmu t_bmu;

//*** La structure de mon réseau de neurons ***

struct str_r
{
  int nv_r;  //*** Le nombre de voisinage (rayon) ***
  t_node **map;
  double *c_vector; //*** Le vecteur courant ***
  double alpha; //*** Le coeif ***
  char *etiq;   //*** L'etiquette ( A,B ou C )***
} str_r;

struct vec
	{
        double *arr;
        char *name;
        double norm;
	};

struct vec * array_vec;

double *aver,*min,*max;

int * index_array;

t_bmu *Bmu = NULL;
int Bmu_size=1;

//***  Initialisations des valeurs ***

void init_conf_B()
{
    conf_B.nbr_n_l=6;
    conf_B.nbr_n_c=10;
    conf_B.T_neuron=conf_B.nbr_n_l*conf_B.nbr_n_c;
    conf_B.v_size=4;
    conf_B.nb_iteration=2500;
    conf_B.minAlpha=0.6;
    conf_B.ftrain=conf_B.nb_iteration/5;
    conf_B.train=2;
}

//*** La fonction pour reading_file iris.data using  strtok ***

void reading_file()
{
    FILE * in;

    char *str=malloc(sizeof(char)*500);
    in=fopen("iris.data","r");

    int i,j;
 for(i=0;i<150;i++)
 {
        fscanf(in,"%s",str);
        char *tok=strtok(str,",");

        for(j=0;j<conf_B.v_size;j++)
            {
                array_vec[i].arr[j]=atof(tok);
                tok=strtok(NULL,",");
            }

        if (strcmp(tok, "Iris-setosa") == 0)
     strcpy(array_vec[i].name,"A");
        else if(strcmp(tok,"Iris-versicolor")==0)
            strcpy(array_vec[i].name,"B");
        else
            strcpy(array_vec[i].name,"C");

        normalize_vector(i,conf_B.v_size);
 }

 fclose(in);
    free(str);
}

//*** La fonction pour allouer ***

void allocate_array_struct(int n)
{
    array_vec=malloc(n*sizeof(struct vec));
    int i;
    for(i=0;i<n;i++)
    {
        array_vec[i].arr=malloc(conf_B.v_size*sizeof(double));
        array_vec[i].name=malloc(20*sizeof(char));
    }
}

//*** Le vecteur moyen ***

void average_vec(int n)
{
    aver=malloc(conf_B.v_size*sizeof(double));
    memset(aver,0,conf_B.v_size*sizeof(double));

    int i,j;

    for(i=0;i<conf_B.v_size;i++)
    {
        for(j=0;j<n;j++)
            aver[i]+=array_vec[j].arr[i];
        aver[i]/=n;
    }
}

//*** Le vecteur min ***

void min_vector(double k)
{
    min=malloc(conf_B.v_size*sizeof(double));
    int i;
    for(i=0;i<conf_B.v_size;i++)
        min[i]=aver[i]-k;
}

//*** Le vecteur max ***

void max_vector(double k)
{
    max=malloc(conf_B.v_size*sizeof(double));
    int i;
    for(i=0;i<conf_B.v_size;i++)
        max[i]=aver[i]+k;
}

//*** La fonction pour normaliser un vecteur ***

void normalize_vector(int i,int size)
{
    double sum=0.;
    int j;
    for(j=0;j<size;j++)
        sum+=(array_vec[i].arr[j])*(array_vec[i].arr[j]);
    array_vec[i].norm=sqrt(sum);
}

//*** Ici afin de  Nous gardons les connexions synaptiques ***

double* init_rand_w()
{
    int i;
    double k=(double)rand()/RAND_MAX;
    double *tmp_w=malloc(conf_B.v_size*sizeof(double));

    for(i=0;i<conf_B.v_size;i++)
        {
            tmp_w[i]=k*(max[i]-min[i])+min[i];
        }

    double norm=0.;

    for(i=0;i<conf_B.v_size;i++)
    {
        norm+=(tmp_w[i])*(tmp_w[i]);
    }

    for(i=0;i<conf_B.v_size;i++)
    {
            tmp_w[i]/=norm;
    }
    return tmp_w;
}

//*** initialisation d'un tableau avant de le mettre en shuffle ***

void init_shuffle(int n)
{
    index_array=malloc(sizeof(int)*n);
    int i;
    for(i=0;i<n;i++)
        index_array[i]=i;
}
//*** Mettre le tableau d'un façon alétoire ***

void array_shuffle(int n)
{
    int i,r_and,k;
    srand(time(NULL));
    for(i=0;i<n;i++)
        {
            r_and=rand()%n;
            k=index_array[i];
            index_array[i]=index_array[r_and];
            index_array[r_and]=k;
        }
}
//*** La fonction pour calculer la distance euclidienne ***

double euclidean_distance(double *a1, double *a2, int n)
{
	double sum=0.0;
	int i;
	for(i=0;i<n;i++)
	{
		sum+=((a1[i] - a2[i])*(a1[i] - a2[i]));
	}
	return sqrt(sum);
} 

//*** La fonction pour calculer le α a chaque fois ***

void calc_alpha(int it_n, int tot_it)
{
	str_r.alpha = conf_B.minAlpha * (1 - ((double)it_n/(double)tot_it)); 
}

//*** Mettre  a jours le nouveau BMU (Best-Matching Unit) ***

void update_bmu(t_bmu* bm_u)
{
    int nr=str_r.nv_r;
    int i,j,x1,x2,y1,y2;

    for(;nr>=0;nr--)
    {
        if(bm_u->r-nr<0)
            x1=0;
        else
            x1=bm_u->r-nr;
        if(bm_u->c-nr<0)
            y1=0;
        else
            y1=bm_u->c-nr;

        if(bm_u->r+nr>conf_B.nbr_n_l-1)
            x2=conf_B.nbr_n_l-1;
        else
            x2=bm_u->r+nr;
        if(bm_u->c+nr>conf_B.nbr_n_c-1)
            y2=conf_B.nbr_n_c-1;
        else
            y2=bm_u->c+nr;

        for(i=x1;i<=x2;i++)
            for(j=y1;j<=y2;j++)
            {

                int k;

                for(k=0;k<conf_B.v_size;k++)
                    {

                        str_r.map[i][j].w[k]+=str_r.alpha*(str_r.c_vector[k]-str_r.map[i][j].w[k]);
                    }
            }
    }
}

//*** La création de ma carte ***

 void create_neuron_map()
{
    int i,j;
    str_r.map=malloc(conf_B.nbr_n_l*sizeof(t_node *));
	for(i=0;i<conf_B.nbr_n_l;i++)
	{
		str_r.map[i]=malloc(conf_B.nbr_n_c*sizeof(t_node));
	}
	for(i=0;i<conf_B.nbr_n_l;i++)
	{
		for (j=0;j<conf_B.nbr_n_c;j++)
		{

                       str_r.map[i][j].w=(double*)malloc(sizeof(double)*conf_B.v_size);
			str_r.map[i][j].w=init_rand_w();
			str_r.map[i][j].etiq=malloc(20*sizeof(char));
			
		}
	}
}

//*** la  fonction pour afficher la map avant et apres le traitement ***

void show_map()
{
    int i,j;
    for(i=0;i<conf_B.nbr_n_l;i++)
    {
        for(j=0;j<conf_B.nbr_n_c;j++)
            {
                printf("%s ",str_r.map[i][j].etiq);
            }
        printf("\n");
    }
}

//*** La phasse d'apprentissage ***

void neurons_learning()
{
    int i,j,p,u,it;
    double min_d,dist;

    Bmu=malloc(sizeof(t_bmu));

    for(p=0;p<conf_B.train;p++)
    {
        int zp;
        if(!p)
        {
            zp=conf_B.ftrain;
        }
        else
        {
            zp=conf_B.nb_iteration-conf_B.ftrain;
            conf_B.minAlpha=0.06;
            str_r.nv_r=1;
        }

        for(it=0;it<zp;it++)
        {
            calc_alpha(it,zp);

            if(it%(conf_B.ftrain/2)==0&&it!=0&&p==0)
            {
                str_r.nv_r-=1;
            }

            array_shuffle(150);

            for(u=0;u<150;u++)
            {
                str_r.c_vector=array_vec[index_array[u]].arr;
                min_d=1000.;
                for(i=0;i<conf_B.nbr_n_l;i++)
                {
                    for(j=0;j<conf_B.nbr_n_c;j++)
                    {
                        dist=euclidean_distance(str_r.c_vector,str_r.map[i][j].w,conf_B.v_size);
                        str_r.map[i][j].act=dist;
                        if(dist<min_d)
                        {
                            min_d=dist;
                            if(Bmu_size>1)
                            {
                                Bmu_size=1;
                                Bmu=realloc(Bmu,Bmu_size*sizeof(t_bmu));
                            }
                            Bmu[0].act=dist;
                            Bmu[0].r=i;
                            Bmu[0].c=j;
                        }
                        else if(dist==min_d)
                        {

                            Bmu_size++;
                            Bmu=realloc(Bmu,Bmu_size*sizeof(t_bmu));
                            Bmu[Bmu_size-1].act=dist;
                            Bmu[Bmu_size-1].r=i;
                            Bmu[Bmu_size-1].c=j;

                        }
                    }
                }

                if(Bmu_size>1)
                {
                    int t=rand()%(Bmu_size);
                    Bmu[0]=Bmu[t];
                }

                strcpy(str_r.map[Bmu[0].r][Bmu[0].c].etiq, array_vec[index_array[u]].name);
                update_bmu(Bmu);
            }
        }
    }
}

//______________________________________________________*** main ***_____________________________________________________________________

   int main()

{
    init_conf_B();

    allocate_array_struct(150);

	  reading_file();

    average_vec(150);
    min_vector(0.02);
    max_vector(0.05);

    init_shuffle(150);

	create_neuron_map();
	
    neurons_learning();
    printf("Résultats l'algorithme SOM data Iris:\n");
    show_map();
    free(aver);
    free(min);
    free(max);
 
	return 0;
}


