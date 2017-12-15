package com.potturi.ai;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;

class KPredict {
    public static void main(String[] args) throws Exception{
        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("/home/spotturi/Scratch/kerasSinModel.json","/home/spotturi/Scratch/kerasSinModel.h5");
        System.out.println(network);
    }
}