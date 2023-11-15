#include <cmath>
#include <vector> 

class AH
{
    public:
        AH(int numX, int numY, double H,double hx,double hy);
        int* Row();
        int* Col();
        double* Val();
        int* row;
        int* col;
        double* val;
        ~AH(); // deconstructor
};
// constuctor 
AH::AH(int numX, int numY, double H,double hx,double hy)
{
    int idx = 0;
    row = new int [numX*numY*9];
    col = new int [numX*numY*9];
    val = new double [numX*numY*9];
    int k = 0;
    for(int j = 0; j < numY; j++){
        for(int i = 0; i < numX; i++){
            int i_n = i - 1;
            int i_p = i + 1;
            int j_n = j - 1;
            int j_p = j + 1;
            double rem = 0.0;

            if ( i == 0 ) {
                i_n = numX - 1;
            }else if ( i == (numX - 1) ){
                i_p = 0;
            }
            if ( j == 0 ){
                j_n = numY - 1;
            }else if ( j == (numY - 1) ){
                j_p = 0;
            }
            k = int(j*numX + i);

            row[idx] = k;
            row[idx+1] = k;
            row[idx+2] = k;
            row[idx+3] = k;
            row[idx+4] = k;
            row[idx+5] = k;
            row[idx+6] = k;
            row[idx+7] = k;
            row[idx+8] = k;
            
            col[idx] = k;
            col[idx + 1] = int(j*numX + i_p);
            col[idx + 2] = int(j*numX + i_n);
            col[idx + 3] = int(j_p*numX + i);
            col[idx + 4] = int(j_n*numX + i);
            col[idx + 5] = int(j_p*numX + i_p);
            col[idx + 6] = int(j_n*numX + i_n);
            col[idx + 7] = int(j_p*numX + i_n);
            col[idx + 8] = int(j_n*numX + i_p);

            val[idx] = -2.0*hy/hx*H - 2.0*hx/hy*H + rem;
            val[idx + 1] = hy/hx*H;
            val[idx + 2] = hy/hx*H;
            val[idx + 3] = hx/hy*H;
            val[idx + 4] = hx/hy*H;
            val[idx + 5] = 1.0/2.0*H;
            val[idx + 6] = 1.0/2.0*H;
            val[idx + 7] = -1.0/2.0*H;
            val[idx + 8] = -1.0/2.0*H;

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
    AH* AH_new(int numX, int numY, double H, double hx, double hy){
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

//g++ -c -fPIC AHb2.cpp -o AHb2.o
//g++ -shared -W1,libAHb2.so -o libAHb2.so AHb2.o
//g++ -shared -o libAHb2.so AHb2.o