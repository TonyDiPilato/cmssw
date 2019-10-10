#ifndef RecoLocalCalo_HGCalRecProducers_HGCalLayerTilesGPU
#define RecoLocalCalo_HGCalRecProducers_HGCalLayerTilesGPU

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"


const float minX_      = hgcaltilesconstants::minX;
const float maxX_      = hgcaltilesconstants::maxX;
const float minY_      = hgcaltilesconstants::minY;
const float maxY_      = hgcaltilesconstants::maxY;
const float minEta_    = hgcaltilesconstants::minEta;
const float maxEta_    = hgcaltilesconstants::maxEta;
const float minPhi_    = hgcaltilesconstants::minPhi;
const float maxPhi_    = hgcaltilesconstants::maxPhi;
const int nColumns_    = hgcaltilesconstants::nColumns;
const int nRows_       = hgcaltilesconstants::nRows;
const int nColumnsEta_ = hgcaltilesconstants::nColumnsEta;
const int nRowsPhi_    = hgcaltilesconstants::nRowsPhi;

const float rX_ = nColumns_/(maxX_-minX_);
const float rY_ = nRows_/(maxY_-minY_);
const float rEta_ = nColumnsEta_/(maxEta_-minEta_);
const float rPhi_ = nRowsPhi_/(maxPhi_-minPhi_);

class HGCalLayerTilesGPU {
  public:
    HGCalLayerTilesGPU() {}


    #ifdef __CUDACC__
    // overload the fill function on device
    __device__
    void fill(float x, float y, float eta, float phi, int isSi, int i)
    {   
      tiles_[getGlobalBin(x,y)].push_back(i);
      if (!isSi) {
        tiles_[getGlobalBinEtaPhi(eta, phi)].push_back(i);
        // Copy cells in phi=[-3.15,-3.] to the last bin
        if (getPhiBin(phi) == mPiPhiBin) {
          tiles_[getGlobalBinEtaPhi(eta, phi + 2 * M_PI)].push_back(i);
        }
        // Copy cells in phi=[3.,3.15] to the first bin
        else if (getPhiBin(phi) == pPiPhiBin) {
          tiles_[getGlobalBinEtaPhi(eta, phi - 2 * M_PI)].push_back(i);
        }
      }
    }
    #endif // __CUDACC__


    __host__ __device__
    int getXBin(float x) const {
      int xBin = (x-minX_)*rX_;
      xBin = (xBin<nColumns_-1 ? xBin:nColumns_-1);
      xBin = (xBin>0 ? xBin:0);
      // cannot use std:clamp
      return xBin;
    }

    __host__ __device__
    int getYBin(float y) const {
      int yBin = (y-minY_)*rY_;
      yBin = (yBin<nRows_-1 ? yBin:nRows_-1);
      yBin = (yBin>0 ? yBin:0);
      // cannot use std:clamp
      return yBin;
    }

    __host__ __device__
    int getEtaBin(float eta) const {
      int etaBin = (eta - minEta_) * rEta_;
      etaBin = (etaBin<nColumnsEta_-1 ? etaBin : nColumnsEta_-1);
      etaBin = (etaBin>0 ? etaBin : 0);
      return etaBin;
    }

    __host__ __device__
    int getPhiBin(float phi) const {
      int phiBin = (phi - minPhi_) * rPhi_;
      phiBin = (phiBin<nRowsPhi_-1 ? phiBin : nRowsPhi_-1);
      phiBin = (phiBin>0 ? phiBin : 0);
      return phiBin;
    }

    int mPiPhiBin = getPhiBin(-M_PI);
    int pPiPhiBin = getPhiBin(M_PI);

    __host__ __device__
    int getGlobalBin(float x, float y) const{
      return getXBin(x) + getYBin(y)*nColumns_;
    }

    __host__ __device__
    int getGlobalBinByBin(int xBin, int yBin) const {
      return xBin + yBin*nColumns_;
    }

    __host__ __device__
    int getGlobalBinEtaPhi(float eta, float phi) const {
      return nColumns_ * nRows_ + getEtaBin(eta) + getPhiBin(phi) * nColumnsEta_;
    }

    __host__ __device__
    int getGlobalBinByBinEtaPhi(int etaBin, int phiBin) const {
      return nColumns_ * nRows_ + etaBin +  phiBin * nColumnsEta_;
    }

    __host__ __device__
    int4 searchBox(float xMin, float xMax, float yMin, float yMax) const {
      return int4{ getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
    }

    __host__ __device__
    int4 searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
      return int4{ getEtaBin(etaMin), getEtaBin(etaMax), getPhiBin(phiMin), getPhiBin(phiMax)};
    }

    __host__ __device__
    void clear() {
      for(auto& t: tiles_) t.reset();
    }

    __host__ __device__
    GPU::VecArray<int, hgcaltilesconstants::maxTileDepth>& operator[](int globalBinId) {
      return tiles_[globalBinId];
    }

  private:
    GPU::VecArray<GPU::VecArray<int, hgcaltilesconstants::maxTileDepth>, hgcaltilesconstants::nColumns * hgcaltilesconstants::nRows > tiles_;

};

  
#endif