/*****************************************************
The code solves numerically the Gradient Descent state evolution (Crisanti-Horner-Sommers-Cugliandolo-Kurchan) equations
with a fixed grid, refer to [Sarao Mannelli, Krzakala, F., & Zdeborova, L. (2019)] for details.
All the integrals are evalutated using the simple 2 points Newton-Cotes formula (coefficient 1/2,1/2).
All the derivatives are solved using the Euler method (f(t+dt)=f(t)+dt*(the rest)).

written: 10/12/18 by Stefano Sarao Mannelli and Pierfrancesco Urbani.
updated: 15/01/19 by Stefano Sarao Mannelli
******************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define PI 3.141592653589793

// This structure contains the physical parameters of the system
typedef struct _psys {
	float p,Delta2,Deltap,T,Cbar0;
} psys;

//	The function computes the equation for mu.
float compute_mu(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h);
//	The function computes the equation for the overlap with the signal. 
float compute_Cbar(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h);
//	The function computes the equation for C.
float compute_C(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h);
//	The function computes the equation for R.
float compute_R(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h);

float print(float** C,float** R,float* mu,float *Cbar,int Nmax,float h,FILE* correlation_file,FILE* response_file,FILE* mu_file,FILE* time_file);
float print_mu(float** C,float** R,float* mu,float *Cbar,int Nmax,float h,FILE* mu_file);

//	The function propagates the initial values of C,R,Cbar in time up to a given time.
int main (int argc, char *argv[]) {
	psys w;
	float h, tmax;
	float Cbar0, Delta2inv;
	float **C, **R, *Cbar, *mu;
	int Nmax,i,j, t,l;
	char suffix[100], directory[100]="";
	char C_file[100], R_file[100], muname_file[100], T_file[100];
	FILE* correlation_file;	FILE* response_file;	FILE* mu_file;	FILE*time_file;

	// Assign the parameters of the system.
	w.p=4.;
	Delta2inv=3.;			// the inverse variance of the matrix channel
	w.Deltap=4.;			// the variance of the tensor channel
	w.Cbar0 = 1e-15;		// initial value of Cbar
	Nmax=1000;				// number of points in the discretization of the grid
	tmax=100.;				// maximum simulation time
	w.Delta2 = 1./Delta2inv;

	h = tmax/((float)Nmax); // grid step size 

	fprintf(stderr, "Dynamics for (2+%.0f)-spin planted with Delta2 %.4f and Deltap %.4f\n",w.p,w.Delta2,w.Deltap);
	fprintf(stderr, "System initialized to %.2e magnetization\n",w.Cbar0);
	fprintf(stderr, "Maximum simulation time %.0f using %d steps of size %.2e\n",tmax,Nmax,h);

    sprintf(suffix, "P%.0f_T0_Delta2inv-%.4f_DeltaP-%.4f_Cbar-%.2e_t-%.1e_Nmax-%d_test",w.p,Delta2inv,w.Deltap,w.Cbar0,tmax,Nmax);
    sprintf(C_file,"%sC_%s.txt",directory,suffix);
    sprintf(R_file,"%sR_%s.txt",directory,suffix);
    sprintf(muname_file,"%smu_%s.txt",directory,suffix);
    sprintf(T_file,"%sT_%s.txt",directory,suffix);

    // Print also correlation, response and Lagrange multiplier.
    // In case of a large grid it can produce large files.
	// correlation_file=fopen(C_file,"w");
	// response_file=fopen(R_file,"w");
	// time_file=fopen(T_file,"w");
	mu_file=fopen(muname_file,"w");

	C=(float**)malloc((Nmax+1)*sizeof(float*));
	R=(float**)malloc((Nmax+1)*sizeof(float*));
	Cbar=(float*)malloc((Nmax+1)*sizeof(float));
	mu=(float*)malloc((Nmax+1)*sizeof(float));
	
	for(i=0; i<=Nmax;i++){  
		
		C[i]=(float*)malloc((i+1)*sizeof(float));
		R[i]=(float*)malloc((i+1)*sizeof(float));
	}

	//inizialization
	C[0][0]=1;
	R[0][0]=0;
	Cbar[0]=w.Cbar0;
	
	// integration loop
	for(i=0; i<=Nmax-1; i++){
		
		t=i;
		compute_mu(mu, Cbar, C, R, t, &w, h);

		C[t+1][t+1]=1;
		R[t+1][t+1]=0;
		
		Cbar[t+1]=compute_Cbar(mu, Cbar, C, R, t, &w, h);

		// compute the two times observables
		for(l=0;l<=t && t<Nmax;l++){
			C[t+1][l]=compute_C(t,l,mu,Cbar,C,R,&w,h);
			R[t+1][l]=compute_R(t,l,mu,Cbar,C,R,&w,h);
		}

		// print progress.
		if(i%100==0){
			fprintf(stderr, "Iteration %d of %d\n",i,Nmax);
		}
		R[t+1][t]=1;
	}
	
	// Print also correlation, response and Lagrange multiplier.
    // print(C,R,mu,Cbar,Nmax,h*w.Delta2,correlation_file,response_file,mu_file,time_file);
	print_mu(C,R,mu,Cbar,Nmax,h,mu_file);
	// fclose(correlation_file);
	// fclose(response_file);
	// fclose(time_file);
	fclose(mu_file);
	free(C);
	free(R);
	free(mu);
		
	return 0;
	
}

//---------------------------------------------------

float compute_mu(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h){
	
	int l;
	float bg;
	float auxp,aux2;
	
	aux2  = C[t][0]*R[t][0];
	auxp  = 0.5*w->p*pow(C[t][0],w->p-1)*R[t][0];
	for(l=1;l<=t-1;l++){
		aux2  += 2.*C[t][l]*R[t][l];
		auxp  += w->p*pow(C[t][l],w->p-1)*R[t][l];
	}
	aux2 += C[t][t]*R[t][t]; aux2/=w->Delta2;
	auxp += 0.5*w->p*pow(C[t][t],w->p-1)*R[t][t]; auxp/=w->Deltap;
	
	mu[t] = pow(Cbar[t],w->p)/w->Deltap+Cbar[t]*Cbar[t]/w->Delta2;
	mu[t]+= h*(auxp+aux2);

	return 0;
	
}

//---------------------------------------------------

float compute_Cbar(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h){

	float Cbar_new;
	float aux2,auxp;
	int m;

	aux2=0.5*R[t][0]*Cbar[0];
	auxp=0.5*(w->p-1)*R[t][0]*pow(C[t][0],w->p-2)*Cbar[0];
	for(m=1;m<=t-1;m++){
		aux2+=R[t][m]*Cbar[m];
		auxp+=(w->p-1)*R[t][m]*pow(C[t][m],w->p-2)*Cbar[m];
	}
	aux2+=0.5*R[t][t]*Cbar[t]; aux2/=w->Delta2;
	auxp+=0.5*(w->p-1)*R[t][t]*pow(C[t][t],w->p-2)*Cbar[t]; auxp/=w->Deltap;

	Cbar_new = Cbar[t]+h*(-mu[t]*Cbar[t]+Cbar[t]/w->Delta2+pow(Cbar[t],w->p-1)/w->Deltap);
	Cbar_new+= h*h*(aux2+auxp);

	return Cbar_new;
}

//-----------------------------------------------------

float compute_C(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h){

	int m,n;
	float auxp1,auxp2,aux21,aux22;
	float Cnew;
	float kroneker;

	if(t==l)kroneker=1;
	else kroneker=0;


	aux21 = .5*R[t][0]*C[l][0];
	auxp1 = .5*(w->p-1)*pow(C[t][0],w->p-2)*R[t][0]*C[l][0];
	// C is filled by the algorithm only when t'<t, in this case we want
	// to swith l and m, so we have to order C.
	for(m=1; m<=l-1;m++){
		aux21 += R[t][m]*C[l][m];
		auxp1 += (w->p-1)*pow(C[t][m],w->p-2)*R[t][m]*C[l][m];
	}
	for(m=l; m<=t-1;m++){		
		aux21 += R[t][m]*C[m][l];
		auxp1 += (w->p-1)*pow(C[t][m],w->p-2)*R[t][m]*C[m][l];
	}
	aux21 += .5*R[t][t]*C[t][l]; aux21/=w->Delta2;
	auxp1 += .5*(w->p-1)*pow(C[t][t],w->p-2)*R[t][t]*C[t][l]; auxp1/=w->Deltap;

	aux22 = .5*C[t][0]*R[l][0];
	auxp2 = .5*pow(C[t][0],w->p-1)*R[l][0];
	for(n=1;n<=l-1;n++){
		aux22 += C[t][n]*R[l][n];
		auxp2 += pow(C[t][n],w->p-1)*R[l][n];
	} 
	aux22 += .5*C[t][l]*R[l][l]; aux22/=w->Delta2;
	auxp2 += .5*pow(C[t][l],w->p-1)*R[l][l]; auxp2/=w->Deltap;

	Cnew = C[t][l]+h*(-mu[t]*C[t][l]+Cbar[l]*Cbar[t]/w->Delta2+Cbar[l]*pow(Cbar[t],w->p-1)/w->Deltap);
	Cnew+= h*((aux21+aux22)*h+(auxp1+auxp2)*h);

	return Cnew;
	
}

//-----------------------------------------------------

float compute_R(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h){

	int m,n;
	float kroneker;
	float auxp,aux2;
	
	float Rnew;

	if(t==l)kroneker=1;
	else kroneker=0;
	
	aux2 = .5*R[t][l]*R[l][l];
	auxp = .5*(w->p-1)*pow(C[t][l],w->p-2)*R[t][l]*R[l][l];
	for(m=l+1; m<=t-1;m++){
		aux2 += R[t][m]*R[m][l];
		auxp += (w->p-1)*pow(C[t][m],w->p-2)*R[t][m]*R[m][l];
	}
	aux2 += .5*R[t][t]*R[t][l]; aux2/=w->Delta2;
	auxp += .5*(w->p-1)*pow(C[t][t],w->p-2)*R[t][t]*R[t][l]; auxp/=w->Deltap;
	
	Rnew = R[t][l]+h*(-mu[t]*R[t][l]+h*aux2+h*auxp);

	return Rnew;
	
}


//---------------------------------------------------

float print(float ** C,float ** R,float *mu,float *Cbar,int Nmax, float h, FILE* correlation_file, FILE* response_file, FILE* mu_file,FILE* time_file){
	
	int i,j;
	
	for(i=0; i<=Nmax;i++){
		
		// if(i%10==0){
		if(1){
			for(j=0;j<Nmax-i; j++){
				
				fprintf(correlation_file, "%f ", C[i+j][j]);
				fprintf(response_file, "%f ", R[i+j][j]);
				fprintf(time_file,"%f", h*(float)(i+j));
			}
			
			fprintf(correlation_file,"\n");
			fprintf(response_file,"\n");
			fprintf(time_file,"\n");
			fprintf(mu_file, "%f \t %f \t %f\n",h*(float)i, Cbar[i], mu[i]);
		}
	}
	
	
	return 1.1;
}

float print_mu(float ** C,float ** R,float *mu,float *Cbar,int Nmax, float h, FILE* mu_file){
	
	int i,j;
	
	for(i=0; i<=Nmax;i++){
		
		// if(i%10==0){
		if(1){
			fprintf(mu_file, "%f \t %f \t %f\n",h*(float)i, Cbar[i], mu[i]);
		}
	}
	
	
	return 1.1;
}