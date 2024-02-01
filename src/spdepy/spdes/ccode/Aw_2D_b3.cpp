#include <cmath>
#include <vector> 


// periodic boundary conditions
class Aw
{
    // Construction of advection matrix of the solution of 2D Advection-Diffsion equation
    // Dirichlet boundary conditions with derivative equal zero one the boundaries
    public:
        Aw(int numX, int numY, const double (*G)[4],double hx,double hy, int diff, const double (*dG)[4]);
        int* Row();
        int* Col();
        double* Val();
        int* row;
        int* col;
        double* val;
        ~Aw(); // deconstructor
};
// constuctor 
Aw::Aw(int numX, int numY, const double (*G)[4],double hx,double hy, int diff, const double (*dG)[4])
{
    int idx = 0;
    row = new int [numX*numY*5];
    col = new int [numX*numY*5];
    val = new double [numX*numY*5];
    int k = 0;
    for(int j = 0; j < numY; j++){
        for(int i = 0; i < numX; i++){
            int i_n = i - 1;
            int i_p = i + 1;
            int j_n = j - 1;
            int j_p = j + 1;

            k = int(j*numX + i);
            
            if (diff == 1){
                val[idx] = (G[k][0]/fabs(G[k][0])*dG[k][0] + dG[k][0] + G[k][2]/fabs(G[k][2])*dG[k][2] - dG[k][2])*hy/2;
                val[idx + 1] = - (G[k][0]/fabs(G[k][0])*dG[k][0] - dG[k][0])*hy/2;
                val[idx + 2] = - (G[k][2]/fabs(G[k][2])*dG[k][2] + dG[k][2])*hy/2;
                val[idx + 3] = 0.0;
                val[idx + 4] = 0.0;
            }else if (diff == 2){
                val[idx] =  (G[k][1]/fabs(G[k][1])*dG[k][1] + dG[k][1] + G[k][3]/fabs(G[k][3])*dG[k][3] - dG[k][3])*hx/2;
                val[idx + 1] = 0.0;
                val[idx + 2] = 0.0;
                val[idx + 3] = - (G[k][1]/fabs(G[k][1])*dG[k][1] - dG[k][1])*hx/2;
                val[idx + 4] = - (G[k][3]/fabs(G[k][3])*dG[k][3] + dG[k][3])*hx/2;
            }else{
                val[idx] = (fabs(G[k][0]) + G[k][0] + fabs(G[k][2]) - G[k][2])*hy/2 + (fabs(G[k][1]) + G[k][1] + fabs(G[k][3]) - G[k][3])*hx/2;
                val[idx + 1] = - (fabs(G[k][0]) - G[k][0])*hy/2;
                val[idx + 2] = - (fabs(G[k][2]) + G[k][2])*hy/2;
                val[idx + 3] = - (fabs(G[k][1]) - G[k][1])*hx/2;
                val[idx + 4] = - (fabs(G[k][3]) + G[k][3])*hx/2;
            }

            
            row[idx] = k;
            if ( i == (numX - 1) ){
                row[idx + 1] = int(numX*numY);
            }else{
                row[idx + 1] = k;
            }
            
            if ( i == 0 ) {
                row[idx + 2] = int(numX*numY);
            }else{
                row[idx + 2] = k;
            }

            if ( j == (numY - 1) ){
                row[idx + 3] = int(numX*numY);
            }else{
                row[idx + 3] = k;
            }

            if ( j == 0 ){
                row[idx + 4] = int(numX*numY);
            }else{
                row[idx + 4] = k;
            }
            
            col[idx] = k;
            col[idx + 1] = int(j*numX + i_p);
            col[idx + 2] = int(j*numX + i_n);
            col[idx + 3] = int(j_p*numX + i);
            col[idx + 4] = int(j_n*numX + i);

            idx += 5;
        }
    }
}

Aw::~Aw(){
    delete[] row;
    delete[] col;
    delete[] val;
}
// get row indices
int* Aw::Row()
{
    return row;
}

// get row indices
int* Aw::Col()
{
    return col;
}

// get row values
double* Aw::Val()
{
    return val;
}

// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
    Aw* Aw_new(int numX, int numY, const double (*G)[4],double hx,double hy, int diff, const double (*dG)[4]){
        return new Aw(numX, numY, G, hx, hy, diff, dG);}
    int* Aw_Row(Aw* aw) {return aw->Row();}
    int* Aw_Col(Aw* aw) {return aw->Col();}
    double* Aw_Val(Aw* aw) {return aw->Val();}
    void Aw_delete(Aw* aw)
    {
        if (aw)
        {
            delete aw;
            aw = nullptr;
        }
    }
}

//g++ -c -fPIC Awb3.cpp -o Awb3.o
//g++ -shared -W1,libAwb3.so -o libAwb3.so Awb3.o
//g++ -shared -o libAwb3.so Awb3.o