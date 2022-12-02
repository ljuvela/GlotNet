#ifndef GLOTNET_LFILTER_H
#define GLOTNET_LFILTER_H


namespace glotnet
{



class AllPoleFilter
{
public:
    AllPoleFilter();
    ~AllPoleFilter();
    void setCoefficients(double *a, double *b, int order);
    void reset();
    double filter(double input);

private:
    double *a;
    double *b;
    double *x;
    double *y;
    int order;
};

} // glotnet


#endif // GLOTNET_LFILTER_H