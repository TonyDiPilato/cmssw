#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTilesGPU.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalCLUEAlgoGPURunner.h"


//GPU Add
#include <math.h>
#include <limits>
#include <iostream>

// for timing
#include <chrono>
#include <ctime>



__device__ float getDeltaCFromLayer(int layer, float delta_c_EE, float delta_c_FH, float delta_c_BH){
  if (layer%maxlayer < lastLayerEE)
    return delta_c_EE;
  else if (layer%maxlayer < lastLayerFH)
    return delta_c_FH;
  else
    return delta_c_BH;
}

__global__ void kernel_compute_histogram( HGCalLayerTilesGPU *d_hist, 
                                          CellsOnLayerPtr d_cells, 
                                          int numberOfCells
                                          )
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < numberOfCells) {
    int layer = d_cells.layer[idx];
    d_hist[layer].fill(d_cells.x[idx], d_cells.y[idx], 
                       d_cells.eta[idx], d_cells.phi[idx], 
                       d_cells.isSi[idx], idx);
  }
  
} //kernel

__global__ void kernel_compute_density( HGCalLayerTilesGPU *d_hist, 
                                        CellsOnLayerPtr d_cells, 
                                        float delta_c_EE, float delta_c_FH, float delta_c_BH,
                                        float delta_r,
                                        int scintMaxIphi_,
                                        bool use2x2_,
                                        int numberOfCells
                                        ) 
{ 
  
  int idxOne = blockIdx.x * blockDim.x + threadIdx.x;
  if (idxOne < numberOfCells){
    double rho{0.};
    int layer = d_cells.layer[idxOne];
    float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);
    
    if (d_cells.isSi[idxOne]) {
      float xOne = d_cells.x[idxOne];
      float yOne = d_cells.y[idxOne];

      // search box with histogram
      int4 search_box = d_hist[layer].searchBox(xOne - delta_c, xOne + delta_c, yOne - delta_c, yOne + delta_c);

      // loop over bins in search box
      for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
        for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {
          int binIndex = d_hist[layer].getGlobalBinByBin(xBin,yBin);
          size_t binSize  = d_hist[layer][binIndex].size();

          // loop over bin contents
          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int idxTwo = d_hist[layer][binIndex][j];
            if (d_cells.isSi[idxTwo]) {  //silicon cells cannot talk to scintillator cells
              const float dist = distance(xOne, yOne, d_cells.x[idxTwo], d_cells.y[idxTwo]);
              if(dist < delta_c) { 
                rho += (idxOne == idxTwo ? 1. : 0.5) * d_cells.weight[idxTwo];              
              }
            }
          }
        }
      }
    } else {
      float etaOne = d_cells.eta[idxOne];
      float phiOne = d_cells.phi[idxOne];
      
      // search box with histogram
      int4 search_box = d_hist[layer].searchBoxEtaPhi(etaOne - delta_r, etaOne + delta_r, phiOne - delta_r, phiOne + delta_r);

      rho += d_cells.weight[idxOne];
      float northeast(0), northwest(0), southeast(0), southwest(0), all(0);

      for (int etaBin = search_box.x; etaBin < search_box.y + 1; ++etaBin) {
        for (int phiBin = search_box.z; phiBin < search_box.w + 1; ++phiBin) {
          int binId = d_hist[layer].getGlobalBinByBinEtaPhi(etaBin, phiBin);
          size_t binSize = d_hist[layer][binId].size();

          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int idxTwo = d_hist[layer][binId][j];
            if (!d_cells.isSi[idxTwo]) {  //scintillator cells cannot talk to silicon cells
              const float dist = distanceEtaPhi(etaOne, phiOne, d_cells.eta[idxTwo], d_cells.phi[idxTwo]);  
              if (dist < delta_r) {
                int iPhiOne = HGCScintillatorDetId(d_cells.detid[idxOne]).iphi();
                int iPhiTwo = HGCScintillatorDetId(d_cells.detid[idxTwo]).iphi();
                int iEtaOne = HGCScintillatorDetId(d_cells.detid[idxOne]).ieta();
                int iEtaTwo = HGCScintillatorDetId(d_cells.detid[idxTwo]).ieta();
                int dIPhi = iPhiTwo - iPhiOne;
                dIPhi += abs(dIPhi) < 2
                             ? 0
                             : dIPhi < 0 ? scintMaxIphi_
                                         : -scintMaxIphi_;  // cells with iPhi=288 and iPhi=1 should be neiboring cells
                int dIEta = iEtaTwo - iEtaOne;

                if (idxTwo != idxOne) {
                  float neighborCellContribution = 0.5f * d_cells.weight[idxTwo];
                  all += neighborCellContribution;
                  if (dIPhi >= 0 && dIEta >= 0)
                    northeast += neighborCellContribution;
                  if (dIPhi <= 0 && dIEta >= 0)
                    southeast += neighborCellContribution;
                  if (dIPhi >= 0 && dIEta <= 0)
                    northwest += neighborCellContribution;
                  if (dIPhi <= 0 && dIEta <= 0)
                    southwest += neighborCellContribution;
                }
              }
            }
          }
        }
      }
      float neighborsval = (std::max(northeast, northwest) > std::max(southeast, southwest))
                               ? std::max(northeast, northwest)
                               : std::max(southeast, southwest);
      if (use2x2_)
        rho += neighborsval;
      else
        rho += all;
    }
    d_cells.rho[idxOne] = rho;
  } // if idx<num_of_cells
} //kernel


__global__ void kernel_compute_distanceToHigher(HGCalLayerTilesGPU* d_hist, 
                                                CellsOnLayerPtr d_cells, 
                                                float delta_c_EE, float delta_c_FH, float delta_c_BH,
                                                float delta_r, float outlierDeltaFactor_, 
                                                int numberOfCells
                                                ) 
{
  int idxOne = blockIdx.x * blockDim.x + threadIdx.x;

  if (idxOne < numberOfCells){
    float maxDelta = std::numeric_limits<float>::max();
    float idxOne_delta = maxDelta;
    int idxOne_nearestHigher = -1;
    float rhoOne = d_cells.rho[idxOne];
    int layer = d_cells.layer[idxOne];

    if (d_cells.isSi[idxOne]) {
      float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);

      float xOne = d_cells.x[idxOne];
      float yOne = d_cells.y[idxOne];
      
      // search box with histogram
      // guarantee to cover a range "outlierDeltaFactor_*delta_c"
      float range = outlierDeltaFactor_ * delta_c;
      int4 search_box = d_hist[layer].searchBox(xOne - range, xOne + range, yOne - range, yOne + range);

      // loop over bins in search box
      for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
        for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {
          size_t binIndex = d_hist[layer].getGlobalBinByBin(xBin,yBin);
          size_t binSize  = d_hist[layer][binIndex].size();

          // loop over bin contents
          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int idxTwo = d_hist[layer][binIndex][j];
            if (d_cells.isSi[idxTwo]){
              const float dist = distance(xOne, yOne, d_cells.x[idxTwo], d_cells.y[idxTwo]);
              const float rhoTwo = d_cells.rho[idxTwo]; 

              // In GPU rhoTwo==rhoOne sometimes is not true when it should be (due to approximations)
              // So we use a epsilon: 1e-7 But the threshold is not sufficiently small
              // Using 1e-8 is too small
              const bool foundHigher = (rhoTwo > rhoOne) || ( rhoTwo == rhoOne && (d_cells.detid[idxTwo] > d_cells.detid[idxOne]));
              const bool distLower = (dist < idxOne_delta) || ((dist == idxOne_delta) && (rhoTwo > d_cells.rho[idxOne_nearestHigher]));
              
              if(foundHigher && distLower) {
                // update idxOne_delta
                idxOne_delta = dist;
                // update idxOne_nearestHigher
                idxOne_nearestHigher = idxTwo;
              }
            }
          }
        }
      } // finish looping over search box
      bool foundNearestHigherInSearchBox = (idxOne_delta != maxDelta);
      // if i is not a seed or noise
      if (foundNearestHigherInSearchBox){
        // pass idxOne_delta and idxOne_nearestHigher to ith hit
        d_cells.delta[idxOne] = idxOne_delta;
        d_cells.nearestHigher[idxOne] = idxOne_nearestHigher;
      } else {
        // otherwise delta is garanteed to be larger outlierDeltaFactor_*delta_c
        // we can safely maximize delta to be maxDelta
        d_cells.delta[idxOne] = maxDelta;
        d_cells.nearestHigher[idxOne] = -1;
      }
    } else { // if is scintillator
      // similar to silicon, but with eta, phi
      const float range = outlierDeltaFactor_ * delta_r;
      const float etaOne = d_cells.eta[idxOne];
      const float phiOne = d_cells.phi[idxOne];
      int4 search_box = d_hist[layer].searchBoxEtaPhi(etaOne - range, etaOne + range, 
                                                      phiOne - range, phiOne + range);
      // loop over all bins in the search box
      for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
        for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {
          // get the id of this bin
          const size_t binIndex = d_hist[layer].getGlobalBinByBinEtaPhi(xBin, yBin);
          // get the size of this bin
          const size_t binSize = d_hist[layer][binIndex].size();

          // loop over all hits in this bin
          for (unsigned int j = 0; j < binSize; j++) {
            unsigned int idxTwo = d_hist[layer][binIndex][j];            
            if (!d_cells.isSi[idxTwo]) {  //scintillator cells cannot talk to silicon cells
              const float dist = distanceEtaPhi(etaOne, phiOne, d_cells.eta[idxTwo], d_cells.phi[idxTwo]);
              const float rhoTwo = d_cells.rho[idxTwo]; 
              const bool foundHigher = (rhoTwo > rhoOne) ||
                                 ((rhoTwo == rhoOne) && (d_cells.detid[idxTwo] > d_cells.detid[idxOne]));
              // if dist == idxOne_delta, then last comer being the nearest higher
              if (foundHigher && ((dist < idxOne_delta) || ((dist == idxOne_delta) && (d_cells.detid[idxTwo] > d_cells.detid[idxOne_nearestHigher])))) {
                // update idxOne_delta
                idxOne_delta = dist;
                // update idxOne_nearestHigher
                idxOne_nearestHigher = idxTwo;
              }
            }
          }
        }
      }

      bool foundNearestHigherInSearchBox = (idxOne_delta != maxDelta);
      if (foundNearestHigherInSearchBox) {
        d_cells.delta[idxOne] = idxOne_delta;
        d_cells.nearestHigher[idxOne] = idxOne_nearestHigher;
      } else {
        // otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
        // we can safely maximize delta to be maxDelta
        d_cells.delta[idxOne] = maxDelta;
        d_cells.nearestHigher[idxOne] = -1;
      }
    } // if isSi
  } // if idx < numOfCells
} //kernel



__global__ void kernel_find_clusters( GPU::VecArray<int,maxNSeeds>* d_seeds,
                                      GPU::VecArray<int,maxNFollowers>* d_followers,
                                      CellsOnLayerPtr d_cells,
                                      float delta_c_EE, float delta_c_FH, float delta_c_BH,
                                      float kappa_, 
                                      float outlierDeltaFactor_,
                                      int numberOfCells
                                      ) 
{
  int idxOne = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idxOne < numberOfCells) {
    int layer = d_cells.layer[idxOne];
    float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);

    float rho_c = kappa_ * d_cells.sigmaNoise[idxOne];

    // initialize clusterIndex
    d_cells.clusterIndex[idxOne] = -1;

    float deltaOne = d_cells.delta[idxOne];
    float rhoOne = d_cells.rho[idxOne];

    bool isSeed = (deltaOne > delta_c) && (rhoOne >= rho_c);
    bool isOutlier = (deltaOne > outlierDeltaFactor_*delta_c) && (rhoOne < rho_c);

    if (isSeed) {
      d_cells.isSeed[idxOne] = 1;
      d_seeds[layer].push_back(idxOne);
    } else {
      if (!isOutlier) {
        int idxOne_NH = d_cells.nearestHigher[idxOne];
        d_followers[idxOne_NH].push_back(idxOne);  
      }
    }
  }
} //kernel

__global__ void kernel_get_n_clusters(GPU::VecArray<int,maxNSeeds>* d_seeds, int* d_nClusters)
{ 
  int idxLayer = threadIdx.x;
  d_nClusters[idxLayer] = d_seeds[idxLayer].size();
}

__global__ void kernel_assign_clusters( GPU::VecArray<int,maxNSeeds>* d_seeds, 
                                        GPU::VecArray<int,maxNFollowers>* d_followers,
                                        CellsOnLayerPtr d_cells,
                                        int * d_nClusters
                                        )
{

  int idxLayer = blockIdx.x;
  int idxCls = blockIdx.y*blockDim.x + threadIdx.x;
  

  if (idxCls < d_nClusters[idxLayer]){

    // buffer is "localStack"
    int buffer[BufferSizePerSeed] = {-1};
    int bufferSize = 0;

    // asgine cluster to seed[idxCls]
    int idxThisSeed = d_seeds[idxLayer][idxCls];
    d_cells.clusterIndex[idxThisSeed] = idxCls;
    // push_back idThisSeed to buffer
    buffer[bufferSize] = idxThisSeed;
    bufferSize ++;

    // process all elements in buffer
    while (bufferSize>0){
      // get last element of buffer
      int idxEndOfBuffer = buffer[bufferSize-1];

      int temp_clusterIndex = d_cells.clusterIndex[idxEndOfBuffer];
              
      // pop_back last element of buffer
      buffer[bufferSize-1] = 0;
      bufferSize--;

      // loop over followers of last element of buffer
      for( int j : d_followers[idxEndOfBuffer] ){
          // pass id to follower
        d_cells.clusterIndex[j] = temp_clusterIndex;
        // push_back follower to buffer
        buffer[bufferSize] = j;
        bufferSize++;
      }
    }
  }
} //kernel



void ClueGPURunner::clueGPU(std::vector<CellsOnLayer> & cells_,
            std::vector<int> & numberOfClustersPerLayer_, 
            std::vector<double> & deltas_,
            double kappa_,
            float outlierDeltaFactor_,
            int scintMaxIphi_,
            bool use2x2_
            ) {
  const int numberOfLayers = cells_.size();
  const float delta_c_EE = deltas_[0];
  const float delta_c_FH = deltas_[1];
  const float delta_c_BH = deltas_[2];
  const float delta_r = deltas_[3];

  //////////////////////////////////////////////
  // copy from cells to local SoA
  // this is fast and takes 3~4 ms on a PU200 event
  //////////////////////////////////////////////
  // auto start1 = std::chrono::high_resolution_clock::now();

  int indexLayerEnd[numberOfLayers];
  // populate local SoA
  CellsOnLayer localSoA;
  for (int i=0; i < numberOfLayers; i++){
    localSoA.detid.insert( localSoA.detid.end(), cells_[i].detid.begin(), cells_[i].detid.end() ); 
    localSoA.isSi.insert( localSoA.isSi.end(), cells_[i].isSi.begin(), cells_[i].isSi.end() );
    localSoA.x.insert( localSoA.x.end(), cells_[i].x.begin(), cells_[i].x.end() );
    localSoA.y.insert( localSoA.y.end(), cells_[i].y.begin(), cells_[i].y.end() );
    localSoA.eta.insert( localSoA.eta.end(), cells_[i].eta.begin(), cells_[i].eta.end() );
    localSoA.phi.insert( localSoA.phi.end(), cells_[i].phi.begin(), cells_[i].phi.end() );
    localSoA.layer.insert( localSoA.layer.end(), cells_[i].layer.begin(), cells_[i].layer.end() );
    localSoA.weight.insert( localSoA.weight.end(), cells_[i].weight.begin(), cells_[i].weight.end() );
    localSoA.sigmaNoise.insert( localSoA.sigmaNoise.end(), cells_[i].sigmaNoise.begin(), cells_[i].sigmaNoise.end() );
    
    int numberOfCellsOnLayer = cells_[i].weight.size();
    if (i == 0){
      indexLayerEnd[i] = -1 + numberOfCellsOnLayer;
    } else {
      indexLayerEnd[i] = indexLayerEnd[i-1] + numberOfCellsOnLayer;
    }
  }  

  const int numberOfCells = indexLayerEnd[numberOfLayers-1] + 1;
  // prepare SoA
  localSoA.rho.resize(numberOfCells,0);
  localSoA.delta.resize(numberOfCells,9999999);
  localSoA.nearestHigher.resize(numberOfCells,-1);
  localSoA.clusterIndex.resize(numberOfCells,-1);
  localSoA.isSeed.resize(numberOfCells,0);
  // auto finish1 = std::chrono::high_resolution_clock::now();

  //////////////////////////////////////////////
  // run on GPU
  //////////////////////////////////////////////
  // auto start2 = std::chrono::high_resolution_clock::now();

  assign_number_of_cells(numberOfCells);
  assign_number_of_layers(numberOfLayers);
  copy_todevice(localSoA);
  clear_set();

  // launch kernels
  const dim3 blockSize(1024,1,1);
  const dim3 gridSize(ceil(numberOfCells/1024.0),1,1);
  kernel_compute_histogram <<<gridSize,blockSize>>>(d_hist, d_cells, numberOfCells);
  kernel_compute_density <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, delta_r, scintMaxIphi_, use2x2_,numberOfCells);
  kernel_compute_distanceToHigher <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, delta_r, outlierDeltaFactor_, numberOfCells);
  kernel_find_clusters <<<gridSize,blockSize>>>(d_seeds, d_followers, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, kappa_, outlierDeltaFactor_, numberOfCells);  
  
  const dim3 blockSize_nlayers(numberOfLayers,1,1);
  const dim3 gridSize_1(1,1,1);
  kernel_get_n_clusters <<<gridSize_1,blockSize_nlayers>>>(d_seeds,d_nClusters);

  const dim3 gridSize_nlayers(numberOfLayers,ceil(maxNSeeds/1024.0),1);
  kernel_assign_clusters <<<gridSize_nlayers,blockSize>>>(d_seeds, d_followers, d_cells, d_nClusters);

  copy_tohost(localSoA,numberOfClustersPerLayer_);
  // auto finish2 = std::chrono::high_resolution_clock::now();

  //////////////////////////////////////////////
  // copy from local SoA to cells 
  // this is fast and takes 1~2 ms on a PU200 event
  //////////////////////////////////////////////
  // auto start3 = std::chrono::high_resolution_clock::now();
  for (int i=0; i < numberOfLayers; i++){
    int numberOfCellsOnLayer = cells_[i].weight.size();
    int indexBegin = indexLayerEnd[i]+1 - numberOfCellsOnLayer;

    cells_[i].rho.resize(numberOfCellsOnLayer);
    cells_[i].delta.resize(numberOfCellsOnLayer);
    cells_[i].nearestHigher.resize(numberOfCellsOnLayer);
    cells_[i].clusterIndex.resize(numberOfCellsOnLayer);
    cells_[i].isSeed.resize(numberOfCellsOnLayer);

    memcpy(cells_[i].rho.data(), &localSoA.rho[indexBegin], sizeof(float)*numberOfCellsOnLayer);
    memcpy(cells_[i].delta.data(), &localSoA.delta[indexBegin], sizeof(float)*numberOfCellsOnLayer);
    memcpy(cells_[i].nearestHigher.data(), &localSoA.nearestHigher[indexBegin], sizeof(int)*numberOfCellsOnLayer);
    memcpy(cells_[i].clusterIndex.data(), &localSoA.clusterIndex[indexBegin], sizeof(int)*numberOfCellsOnLayer); 
    memcpy(cells_[i].isSeed.data(), &localSoA.isSeed[indexBegin], sizeof(int)*numberOfCellsOnLayer);
  }

  // auto finish3 = std::chrono::high_resolution_clock::now();
  // std::cout << (std::chrono::duration<double>(finish1-start1)).count() << "," 
  //           << (std::chrono::duration<double>(finish2-start2)).count() << ","
  //           << (std::chrono::duration<double>(finish3-start3)).count() << ",";

}



