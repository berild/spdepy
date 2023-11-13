
#include <cmath>
#include <vector> 


// periodic boundary conditions
class Aw
{
    public:
        Aw(int numX, int numY, double G[2],double hx,double hy);
        int* Row();
        int* Col();
        double* Val();
        int* row;
        int* col;
        double* val;
        ~Aw(); // deconstructor
};
// constuctor 
Aw::Aw(int numX, int numY, double G[2],double hx,double hy)
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
            double rem = 0.0;

            k = int(j*numX + i);
            
            val[idx] = (fabs(G[0]) + G[0] + fabs(G[2]) - G[2])*hy/2 + (fabs(G[1]) + G[1] + fabs(G[3]) - G[3])*hx/2;
            val[idx + 1] = - (fabs(G[0]) - G[0])*hy/2;
            val[idx + 2] = - (fabs(G[2]) + G[2])*hy/2;
            val[idx + 3] = - (fabs(G[1]) - G[1])*hx/2;
            val[idx + 4] = - (fabs(G[3]) + G[3])*hx/2;

            if ( i == 0 ) {
                i_n = numX-1;
                val[idx + 2] = 0.0;
            }else if ( i == (numX - 1) ){
                i_p = 0;
                val[idx + 1] = 0.0;
            }
            if ( j == 0 ){
                j_n = numY - 1;
                val[idx + 4] = 0.0;
            }else if ( j == (numY - 1) ){
                j_p = 0;
                val[idx + 3] = 0.0;
            }

            row[idx] = k;
            row[idx + 1] = k;
            row[idx + 2] = k;
            row[idx + 3] = k;
            row[idx + 4] = k;
            
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
    Aw* Aw_new(int numX, int numY, double G[2],double hx,double hy){
        return new Aw(numX, numY, G,hx,hy);}
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