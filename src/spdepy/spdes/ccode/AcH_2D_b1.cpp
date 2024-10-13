#include <cmath>
#include <vector> 

class AH
{
    // Construction of diffusion matrix of the solution of 2D Advection-Diffsion equation
    // Neumann boundary conditions with derivative equal zero one the boundaries
    public:
        AH(int numX, int numY, double H[2][2],double hx,double hy);
        int* Row();
        int* Col();
        double* Val();
        int* row;
        int* col;
        double* val;
        ~AH(); // deconstructor
};
// constuctor 
AH::AH(int numX, int numY, double H[2][2],double hx,double hy)
{   
    int idx = 0;
    row = new int [numX*numY*9];
    col = new int [numX*numY*9];
    val = new double [numX*numY*9];
    int k = 0;
    double H000;
    double H100;
    double H010;
    double H110;
    double H201;
    double H301;
    double H211;
    double H311;
    for(int j = 0; j < numY; j++){
        for(int i = 0; i < numX; i++){
            int i_n = i - 1;
            int i_p = i + 1;
            int j_n = j - 1;
            int j_p = j + 1;
            double rem = 0.0;
            
            H000 = H[0][0];
            H100 = H[0][0];
            H010 = H[1][0];
            H110 = H[1][0];
            H201 = H[1][0];
            H301 = H[1][0];
            H211 = H[1][1];
            H311 = H[1][1];

            if ( i == 0 ) {
                i_n = i;
                H000 = 0.0;
                H010 = 0.0;
            }else if ( i == (numX - 1) ){
                i_p = i;
                H100 = 0.0;
                H110 = 0.0;
            }
            if ( j == 0 ){
                j_n = j;
                H211 = 0.0;
                H201 = 0.0;
            }else if ( j == (numY - 1) ){
                j_p = j;
                H311 = 0.0;
                H301 = 0.0;
            }
            k = int(j*numX + i);

            if (k == int(j*numX + i_p)){
                rem = rem + hy/hx*H100 + 1.0/4.0*(H301 - H201);
                row[idx + 1] = int(numX*numY);
            }else{
                row[idx + 1] = k;
            }
            if (k == int(j*numX + i_n)){
                rem = rem + hy/hx*H000 - 1.0/4.0*(H301 - H201);
                row[idx + 2] = int(numX*numY);
            }else{
                row[idx + 2] = k;
            }
            if (k == int(j_p*numX + i)){
                rem = rem + hx/hy*H311 + 1.0/4.0*(H110 - H010);
                row[idx + 3] = int(numX*numY);
            }else{
                row[idx + 3] = k;
            }
            if (k == int(j_n*numX + i)){
                rem = rem + hx/hy*H211 - 1.0/4.0*(H110 - H010);
                row[idx + 4] = int(numX*numY);
            }else{
                row[idx + 4] = k;
            }
            if (k == int(j_p*numX + i_p)){
                rem = rem + 1.0/4.0*(H301 + H110);
                row[idx + 5] = int(numX*numY);
            }else{
                row[idx + 5] = k;
            }
            if (k == int(j_n*numX + i_n)){
                rem = rem + 1.0/4.0*(H201 + H010);
                row[idx + 6] = int(numX*numY);
            }else{
                row[idx + 6] = k;
            }
            if (k == int(j_p*numX + i_n)){
                rem = rem - 1.0/4.0*(H301 + H010);
                row[idx + 7] = int(numX*numY);
            }else{
                row[idx + 7] = k;
            }
            if (k == int(j_n*numX + i_p)){
                rem = rem - 1.0/4.0*(H201 + H110);
                row[idx + 8] = int(numX*numY);
            }else{
                row[idx + 8] = k;
            }

            row[idx] = k;

            col[idx] = k;
            col[idx + 1] = int(j*numX + i_p);
            col[idx + 2] = int(j*numX + i_n);
            col[idx + 3] = int(j_p*numX + i);
            col[idx + 4] = int(j_n*numX + i);
            col[idx + 5] = int(j_p*numX + i_p);
            col[idx + 6] = int(j_n*numX + i_n);
            col[idx + 7] = int(j_p*numX + i_n);
            col[idx + 8] = int(j_n*numX + i_p);

            val[idx] = -hy/hx*(H100 + H000)  - hx/hy*(H311 + H211)+ rem;
            val[idx + 1] = hy/hx*H100 + 1.0/4.0*(H301 - H201);
            val[idx + 2] = hy/hx*H000 - 1.0/4.0*(H301 - H201);
            val[idx + 3] = hx/hy*H311 + 1.0/4.0*(H110 - H010);
            val[idx + 4] = hx/hy*H211 - 1.0/4.0*(H110 - H010);
            val[idx + 5] = 1.0/4.0*(H301 + H110);
            val[idx + 6] = 1.0/4.0*(H201 + H010);
            val[idx + 7] = -1.0/4.0*(H301 + H010);
            val[idx + 8] = -1.0/4.0*(H201 + H110);
            idx += 9;
        }
    }
}

AH::~AH(){
    delete[] row;
    delete[] col;
    delete[] val;
}
// get row indices
int* AH::Row()
{
    return row;
}

// get row indices
int* AH::Col()
{
    return col;
}

// get row values
double* AH::Val()
{
    return val;
}

// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
    AH* AH_new(int numX, int numY, double H[2][2],double hx,double hy){
        return new AH(numX, numY, H,hx,hy);}
    int* AH_Row(AH* ah) {return ah->Row();}
    int* AH_Col(AH* ah) {return ah->Col();}
    double* AH_Val(AH* ah) {return ah->Val();}
    void AH_delete(AH* ah)
    {
        if (ah)
        {
            delete ah;
            ah = nullptr;
        }
    }
}

//g++ -c -fPIC AHb1.cpp -o AHb1.o
//g++ -shared -W1,libAHb1.so -o libAHb1.so AHb1.o
//g++ -shared -o libAHb1.so AHb1.o