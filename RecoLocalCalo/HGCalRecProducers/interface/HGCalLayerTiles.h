// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles
#define RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

class HGCalLayerTiles {
public:
  void fill(const std::vector<float>& x,
            const std::vector<float>& y,
            const std::vector<float>& eta,
            const std::vector<float>& phi,
            const std::vector<int>& isSi) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      tiles_[getGlobalBin(x[i], y[i])].push_back(i);
      if (!isSi[i]) {
        tiles_[getGlobalBinEtaPhi(eta[i], phi[i])].push_back(i);
        // Copy cells in phi=[-3.15,-3.] to the last bin
        if (getPhiBin(phi[i]) == mPiPhiBin) {
          tiles_[getGlobalBinEtaPhi(eta[i], phi[i] + 2 * M_PI)].push_back(i);
        }
        // Copy cells in phi=[3.,3.15] to the first bin
        if (getPhiBin(phi[i]) == pPiPhiBin) {
          tiles_[getGlobalBinEtaPhi(eta[i], phi[i] - 2 * M_PI)].push_back(i);
        }
      }
    }
  }

  int getXBin(float x) const {
    constexpr float xRange = hgcaltilesconstants::maxX - hgcaltilesconstants::minX;
    static_assert(xRange >= 0.);
    constexpr float r = hgcaltilesconstants::nColumns / xRange;
    int xBin = (x - hgcaltilesconstants::minX) * r;
    // xBin = std::clamp(xBin, 0, hgcaltilesconstants::nColumns);
    xBin = std::min(xBin,hgcaltilesconstants::nColumns-1);
    xBin = std::max(xBin,0);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = hgcaltilesconstants::maxY - hgcaltilesconstants::minY;
    static_assert(yRange >= 0.);
    constexpr float r = hgcaltilesconstants::nRows / yRange;
    int yBin = (y - hgcaltilesconstants::minY) * r;
    // yBin = std::clamp(yBin, 0, hgcaltilesconstants::nRows);
    yBin = std::min(yBin,hgcaltilesconstants::nRows-1);
    yBin = std::max(yBin,0);
    return yBin;
  }

  int getEtaBin(float eta) const {
    constexpr float etaRange = hgcaltilesconstants::maxEta - hgcaltilesconstants::minEta;
    static_assert(etaRange >= 0.);
    constexpr float r = hgcaltilesconstants::nColumnsEta / etaRange;
    int etaBin = (eta - hgcaltilesconstants::minEta) * r;
    // etaBin = std::clamp(etaBin, 0, hgcaltilesconstants::nColumnsEta);
    etaBin = std::min(etaBin,hgcaltilesconstants::nColumnsEta-1);
    etaBin = std::max(etaBin,0);
    return etaBin;
  }

  int getPhiBin(float phi) const {
    constexpr float phiRange = hgcaltilesconstants::maxPhi - hgcaltilesconstants::minPhi;
    static_assert(phiRange >= 0.);
    constexpr float r = hgcaltilesconstants::nRowsPhi / phiRange;
    int phiBin = (phi - hgcaltilesconstants::minPhi) * r;
    // phiBin = std::clamp(phiBin, 0, hgcaltilesconstants::nRowsPhi);
    phiBin = std::min(phiBin,hgcaltilesconstants::nRowsPhi-1);
    phiBin = std::max(phiBin,0);
    return phiBin;
  }

  int mPiPhiBin = getPhiBin(-M_PI);
  int pPiPhiBin = getPhiBin(M_PI);

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * hgcaltilesconstants::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * hgcaltilesconstants::nColumns; }

  int getGlobalBinEtaPhi(float eta, float phi) const {
    return hgcaltilesconstants::nColumns * hgcaltilesconstants::nRows + getEtaBin(eta) +
           getPhiBin(phi) * hgcaltilesconstants::nColumnsEta;
  }

  int getGlobalBinByBinEtaPhi(int etaBin, int phiBin) const {
    return hgcaltilesconstants::nColumns * hgcaltilesconstants::nRows + etaBin +
           phiBin * hgcaltilesconstants::nColumnsEta;
  }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) const {
    int xBinMin = getXBin(xMin);
    int xBinMax = getXBin(xMax);
    int yBinMin = getYBin(yMin);
    int yBinMax = getYBin(yMax);
    return std::array<int, 4>({{xBinMin, xBinMax, yBinMin, yBinMax}});
  }

  std::array<int, 4> searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
    int etaBinMin = getEtaBin(etaMin);
    int etaBinMax = getEtaBin(etaMax);
    int phiBinMin = getPhiBin(phiMin);
    int phiBinMax = getPhiBin(phiMax);
    return std::array<int, 4>({{etaBinMin, etaBinMax, phiBinMin, phiBinMax}});
  }

  void clear() {
    for (auto& t : tiles_)
      t.clear();
  }

  const std::vector<int>& operator[](int globalBinId) const { return tiles_[globalBinId]; }

private:
  std::array<std::vector<int>, hgcaltilesconstants::nTiles> tiles_;
};

struct CellsOnLayer {
  std::vector<DetId> detid;
  std::vector<int> isSi;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<int> layer;
  std::vector<float> weight;
  std::vector<float> sigmaNoise;
  
  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  // why use int instead of bool?
  // https://en.cppreference.com/w/cpp/container/vector_bool
  // std::vector<bool> behaves similarly to std::vector, but in order to be space efficient, it:
  // Does not necessarily store its elements as a contiguous array (so &v[0] + n != &v[n])


  void clear() {
    detid.clear();
    isSi.clear();
    x.clear();
    y.clear();
    eta.clear();
    phi.clear();
    layer.clear();
    weight.clear();
    rho.clear();
    delta.clear();
    nearestHigher.clear();
    clusterIndex.clear();
    sigmaNoise.clear();
    followers.clear();
    isSeed.clear();
  }
};

#endif
