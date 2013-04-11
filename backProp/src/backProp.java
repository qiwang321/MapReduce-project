import java.io.*;
import java.util.Random;
/**
 * implementing RPROP (batch version)
 * @author qwang37
 *
 */

public class backProp {
	public static float[][] lr;
	public static float[][] br;
	public static int numLayers;
	public static int[] nodesLayer;
	public static float[][] weights;
	public static float[][] weightsInc;
	public static float[][] lastInc; // weight increment for the last time (used for RPROP)
	public static float[][] b;
	public static float[][] bInc;
	public static float[][] bLastInc; // bias increment for the last time
	public static float[][] act; // activation level at each layer
	public static float[][] delta;
	public static Random generator;

	// learning parameter
	private static class Arg {
		public float yitaMax;
		public float yitaMin;
		public int numEpochs;
		public float yitaPlus;
		public float yitaMinus;
	}


	private static void init() {
		generator = new Random(System.nanoTime());

		numLayers = 2;
		lr = new float[numLayers+1][];
		br = new float[numLayers+1][];
		nodesLayer = new int[numLayers+1];
		weights = new float[numLayers+1][];
		weightsInc = new float[numLayers+1][];
		lastInc = new float[numLayers+1][];
		b = new float[numLayers+1][];
		bInc = new float[numLayers+1][];
		bLastInc = new float[numLayers+1][];
		act = new float[numLayers+1][];
		delta = new float[numLayers+1][];

		nodesLayer[0] = 2; // number of input nodes 2
		nodesLayer[1] = 10;
		nodesLayer[2] = 1;
		for (int k = 1; k <=numLayers; k++) {
			lr[k] = new float[nodesLayer[k] * nodesLayer[k-1]];
			br[k] = new float[nodesLayer[k]];
			act[k] = new float[nodesLayer[k]];
			delta[k] = new float[nodesLayer[k]];
			
			weights[k] = new float[nodesLayer[k] * nodesLayer[k-1]];
			weightsInc[k] = new float[nodesLayer[k] * nodesLayer[k-1]];
			lastInc[k] = new float[nodesLayer[k] * nodesLayer[k-1]];
			
			b[k] = new float[nodesLayer[k]];
			bInc[k] = new float[nodesLayer[k]];
			bLastInc[k] = new float[nodesLayer[k]];

			for (int i = 0; i < nodesLayer[k]; i++) {
				act[k][i] = 0.0f;
				delta[k][i] = 0.0f;
				b[k][i] = (float)generator.nextGaussian();
				//b[k][i] = 0.0f;
				br[k][i] = 0.05f; // learning rate for the biases
				//br[k][i] = 0;
				bLastInc[k][i] = 0.0f;
			}
			for (int i = 0; i < nodesLayer[k]*nodesLayer[k-1]; i++) {
				weights[k][i] = (float)generator.nextGaussian();	
				//weights[k][i] = 0.0f;	
				lastInc[k][i] = 0.0f;
				lr[k][i] = 1.0f; // learning rate for the weights
				//lr[k][i] = 0;
			}
		}
	}

	public static void sigm(float[] h, float[] w, float[] b, int n, int m, float[] v) {
		for (int j = 0; j < n; j++) {
			h[j] = -b[j];
			for (int i = 0; i < m; i++) {
				h[j] = h[j] - w[m*j + i] * v[i];
			}
			h[j] = 1.0f / (1 + (float) Math.exp(h[j]));
		}
	}

	public static void train(float[][] data, float[][] target, Arg args) {
	
		for (int epoch = 0; epoch < args.numEpochs; epoch++) {
			// reset the weight increase at the beginning of each epoch
			for (int k = 1; k <= numLayers; k++) {
				for (int i = 0; i < nodesLayer[k]*nodesLayer[k-1]; i++)
					weightsInc[k][i] = 0.0f;
				for (int i = 0; i < nodesLayer[k]; i++)
					bInc[k][i] = 0.0f;
			}
			for (int p = 0; p < data.length; p++) {
				// get all activation values
				act[0] = data[p];
				for (int k = 1; k <= numLayers; k++) {
					sigm(act[k], weights[k], b[k], nodesLayer[k], nodesLayer[k-1], act[k-1]);
				}

				// delta value at the top layer
				for (int j = 0; j < nodesLayer[numLayers]; j++)
					delta[numLayers][j] = (target[p][j] - act[numLayers][j]) * act[numLayers][j] * (1 - act[numLayers][j]); 

				// weight update at the top layer(batch version)
				for (int j = 0; j < nodesLayer[numLayers]; j++) {
					// weight
					for (int i = 0; i < nodesLayer[numLayers-1]; i++) {							
						weightsInc[numLayers][j*nodesLayer[numLayers-1] + i] = weightsInc[numLayers][j*nodesLayer[numLayers-1] + i] + 
								delta[numLayers][j] * act[numLayers-1][i];
					}
					// b
					bInc[numLayers][j] = bInc[numLayers][j] + delta[numLayers][j];
				}

				// back propagation and weight update
				for (int k = numLayers-1; k > 0; k--) {
					// delta propagation
					for (int i = 0; i < nodesLayer[k]; i++) {
						delta[k][i] = 0;
						for (int j = 0; j < nodesLayer[k+1]; j++) {
							delta[k][i] = delta[k][i] + delta[k+1][j] * weights[k+1][j * nodesLayer[k] + i];
						}
					}
					// weight update (batch version)
					for (int j = 0; j < nodesLayer[k]; j++) {
						// weight
						for (int i = 0; i < nodesLayer[k-1]; i++) {							
							weightsInc[k][j*nodesLayer[k-1] + i] = weightsInc[k][j*nodesLayer[k-1] + i] + delta[k][j] * act[k-1][i];
						}
						//b
						bInc[k][j] = bInc[k][j] + delta[k][j];
					}
				}
			}

			// three cases for update
			for (int k = 1; k <= numLayers; k++) {
				// weights
				for (int i = 0; i < nodesLayer[k] * nodesLayer[k-1]; i++)  {
					if (lastInc[k][i] * weightsInc[k][i] > 0) {
						lr[k][i] = Math.min(lr[k][i]*args.yitaPlus, args.yitaMax);
						lastInc[k][i] = lr[k][i] * Math.signum(weightsInc[k][i]);
						weights[k][i] = weights[k][i] + lastInc[k][i];
					}					
					else if (lastInc[k][i] * weightsInc[k][i] < 0) {
						lr[k][i] = Math.max(lr[k][i]*args.yitaMinus, args.yitaMin);
						weights[k][i] = weights[k][i] - lastInc[k][i];
						lastInc[k][i] = 0;
					}
					else {
						lastInc[k][i] = lr[k][i] * Math.signum(weightsInc[k][i]);
						weights[k][i] = weights[k][i] + lastInc[k][i];
					}
				}
				// b
				for (int i = 0; i < nodesLayer[k]; i++) {
					if (bLastInc[k][i] * bInc[k][i] > 0) {
						br[k][i] = Math.min(br[k][i]*args.yitaPlus, args.yitaMax);
						bLastInc[k][i] = br[k][i] * Math.signum(bInc[k][i]);
						b[k][i] = b[k][i] + bLastInc[k][i];
					}					
					else if (lastInc[k][i] * weightsInc[k][i] < 0) {
						br[k][i] = Math.max(br[k][i]*args.yitaMinus, args.yitaMin);
						b[k][i] = b[k][i] - bLastInc[k][i];
						bLastInc[k][i] = 0.0f;
					}
					else {
						bLastInc[k][i] = br[k][i] * Math.signum(bInc[k][i]);
						b[k][i] = b[k][i] + bLastInc[k][i];
					}
				}
			}
		}
	}
	
	
	public static void main(String[] args) throws IOException {
		
		// prepare data
		float data[][]= new float[4 * 1000][2];
		float target[][] = new float[data.length][1];
		
		for (int i = 0; i < 1000; i++) {
			FileInputStream in1 = new FileInputStream("xor");
			BufferedReader din = new BufferedReader(new InputStreamReader(in1));
			FileInputStream in2 = new FileInputStream("xor_target");
			BufferedReader tin = new BufferedReader(new InputStreamReader(in2));
			for (int t = 0; t < 4; t++){
				String[] tmp = din.readLine().split("\\s+");
				String[] tmpTarget = tin.readLine().split("\\s+");
				for (int l = 0; l < data[0].length; l++)
					data[i*4 + t][l] = Float.parseFloat(tmp[l]);
				for (int l = 0; l < target[0].length; l++)
					target[i*4 + t][l] = Float.parseFloat(tmpTarget[l]);
			}
			din.close();
			tin.close();
		}
		
		// train
		init();
		Arg trainArgs = new Arg();
		trainArgs.numEpochs = 2000;
		trainArgs.yitaMax = 10.0f;
		trainArgs.yitaMin = 1e-6f;
		trainArgs.yitaPlus = 1.01f;
		trainArgs.yitaMinus = 0.99f;
		train(data, target, trainArgs);
		
		// test
		FileReader in1 = new FileReader("xor");
		BufferedReader din = new BufferedReader(in1);
		FileReader in2 = new FileReader("xor_target");
		BufferedReader tin = new BufferedReader(in2);
		FileWriter out = new FileWriter("xor-out");
		BufferedWriter tout = new BufferedWriter(out);
		for (int t = 0; t < 4; t++) {
			String[] tmp = din.readLine().split("\\s+");
			String[] tmpTarget = tin.readLine().split("\\s+");
			float[] t0 = new float[1];
			
			for (int l = 0; l < act[0].length; l++)
				act[0][l] = Float.parseFloat(tmp[l]);
			for (int l = 0; l < t0.length; l++)
				t0[l] = Float.parseFloat(tmpTarget[l]);
			
			for (int k = 1; k <= numLayers; k++) {
				sigm(act[k], weights[k], b[k], nodesLayer[k], nodesLayer[k-1], act[k-1]);
			}
			
			for (int l = 0; l < t0.length; l++)	
				tout.write(act[numLayers][l] + " ");
			tout.write(",\t");
			for (int l = 0; l < t0.length; l++)
				tout.write(Float.toString(t0[l]));
			tout.write("\n");
		}
		tout.close();
	}
}








