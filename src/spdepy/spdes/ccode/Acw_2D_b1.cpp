#include <cmath>
#include <vector> 

// reflecting boundary conditions (neumann zero)
class Aw
{
    // Construction of advection matrix of the solution of 2D Advection-Diffsion equation
    // Neumann boundary conditions with derivative equal zero one the boundaries
    public:
        Aw(int numX, int numY, double G[2],double hx,double hy,int diff);
        int* Row();
        int* Col();
        double* Val();
        int* row;
        int* col;
        double* val;
        ~Aw(); // deconstructor
};
// constuctor 
Aw::Aw(int numX, int numY, double G[2],double hx,double hy,int diff)
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


            if (diff == 1){
                val[idx] = G[0]/fabs(G[0])*hy;
                val[idx + 1] = - (G[0]/fabs(G[0]) - 1.0)*hy/2;    
                val[idx + 2] = - (G[0]/fabs(G[0]) + 1.0)*hy/2;
                val[idx + 3] = 0.0;
                val[idx + 4] = 0.0;
            }else if (diff == 2){
                val[idx] = G[1]/fabs(G[1])*hx;
                val[idx + 1] = 0.0;
                val[idx + 2] = 0.0;
                val[idx + 3] = - (G[1]/fabs(G[1]) - 1.0)*hx/2;
                val[idx + 4] = - (G[1]/fabs(G[1]) + 1.0)*hx/2;
            }else{
                val[idx] = fabs(G[0])*hy + fabs(G[1])*hx;
                val[idx + 1] = - (fabs(G[0]) - G[0])*hy/2;
                val[idx + 2] = - (fabs(G[0]) + G[0])*hy/2;
                val[idx + 3] = - (fabs(G[1]) - G[1])*hx/2;
                val[idx + 4] = - (fabs(G[1]) + G[1])*hx/2;
            }

            k = int(j*numX + i);

            row[idx] = k;
            row[idx + 1] = k;
            row[idx + 2] = k;
            row[idx + 3] = k;
            row[idx + 4] = k;

            if ( i == 0 ) {
                i_n = i;
                val[idx] -= fabs(G[0])*hy/2;
                row[idx + 2] = int(numX*numY);
            }else if ( i == (numX - 1) ){
                i_p = i;
                val[idx] -= fabs(G[0])*hy/2;
                row[idx + 1] = int(numX*numY);
            }
            if ( j == 0 ){
                j_n = j;
                val[idx] -= fabs(G[1])*hx/2;
                row[idx + 4] = int(numX*numY);
            }else if ( j == (numY - 1) ){
                j_p = j;
                val[idx] -= fabs(G[1])*hx/2;
                row[idx + 3] = int(numX*numY);
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
    Aw* Aw_new(int numX, int numY, double G[2],double hx,double hy, int diff){
        return new Aw(numX, numY, G,hx,hy,diff);}
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

//g++ -c -fPIC Awb1.cpp -o Awb1.o
//g++ -shared -W1,libAwb1.so -o libAwb1.so Awb1.o
//g++ -shared -o libAwb1.so Awb1.o





            // if (k == int(j*numX + i_p)){
            //     // val[idx] += val[idx + 1];
            // }else{
            // }
            // if (k == int(j*numX + i_n)){
            //     // val[idx] += val[idx + 2];
            // }else{
            // }
            // if (k == int(j_p*numX + i)){
            //     // val[idx] += val[idx + 3];
            //     val[idx] -= fabs(G[1])*hx/2;
            //     row[idx + 3] = int(numX*numY);
            // }else{
            //     row[idx + 3] = k;
            // }
            // if (k == int(j_n*numX + i)){
            //     // val[idx] += val[idx + 4];
            //     val[idx] -= fabs(G[1])*hx/2;
            //     row[idx + 4] = int(numX*numY);
            // }else{
            //     row[idx + 4] = k;
            // }