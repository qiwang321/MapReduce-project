package model;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.Writable;



public class SuperModel1 implements Writable {
	private ArrayList<ModelNode1> models;
	
	// dimension parameters
	private int nModel;
	//private int nInput; // number of input nodes for the super layer
	private int nEach; // the number of outputs for each sub model
	private int NUM_LAYER = GlobalUtil1.SUPER_NUM;
	private int NODES_INPUT; // different from GlobalUtil1!    
	private int sizeInput = GlobalUtil1.sizeInput;
	//private final int[] train_len = GlobalUtil1.train_len; 
	//private final int[] test_len = GlobalUtil1.test_len;
	private int[] nodes_layer = GlobalUtil1.super_layer;

	private final Random rd = new Random();  
	private float[][] sample_mem = new float[NUM_LAYER+1][]; //space storing the MCMC samples
	private float[][] weights = new float[NUM_LAYER+1][]; //space storing the updating weights (first is not used)
	private float[][] bh = new float[NUM_LAYER+1][]; // hidden layer biases (rbm)
	private float[][] bv = new float[NUM_LAYER+1][]; // visible layer biases (rbm)  

	// training parameters
	private float yita_w = GlobalUtil1.yita_w, yita_bv = GlobalUtil1.yita_bv, yita_bh = GlobalUtil1.yita_bh,
			yita_wt = GlobalUtil1.yita_wt, yita_bvt = GlobalUtil1.yita_bvt, yita_bht = GlobalUtil1.yita_bht; // learning rates
	private float mu = GlobalUtil1.mu, reg = GlobalUtil1.reg;

	public SuperModel1(ArrayList<ModelNode1> m) {
		models = m;
		nModel = m.size();
		nEach = GlobalUtil1.nodes_layer[GlobalUtil1.NUM_LAYER];
		NODES_INPUT = nModel * nEach;
		nodes_layer = GlobalUtil1.super_layer;
		nodes_layer[0] = NODES_INPUT;
		sample_mem[0] = new float[NODES_INPUT];
		for (int k = 1; k <= NUM_LAYER; k++) {
			weights[k] = new float[nodes_layer[k] * nodes_layer[k-1]];
			for (int i = 0; i < nodes_layer[k] * nodes_layer[k-1]; i++)
				weights[k][i] = 0.1f * (float)rd.nextGaussian();
			sample_mem[k] = new float[nodes_layer[k]];
			bh[k] = new float[nodes_layer[k]];
			bv[k] = new float[nodes_layer[k-1]];
		}

	}

	public void train(BufferedReader in) throws IOException {
		// readin the data
		float[] data = new float[GlobalUtil1.sizeInput]; // place to hold float data
		String line = in.readLine();
		while (line != null && line.length() > 0) {
			String[] tmp = line.split("\\s+");
			for (int i = 0; i < GlobalUtil1.sizeInput; i++) {
				data[i] = Float.parseFloat(tmp[i]) / 255.0f;
			}

			// simulate each submodel
			for (int i = 0; i < nModel; i++) {
				float[] sub_top = models.get(i).sim(data);
				for (int j = 0; j < nEach; j++)
					sample_mem[0][i*nEach+j] = sub_top[j];
			}
			
			// acquired data for super model, standard training
			for (int layer_ind = 1; layer_ind <= NUM_LAYER; layer_ind++)
				work_update(layer_ind);

			line = in.readLine();
		}
	}


	@Override
	public void readFields(DataInput arg0) throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public void write(DataOutput arg0) throws IOException {
		// TODO Auto-generated method stub

	}
	
	@Override 
	
	public String toString(){return "finished";}
	/*
	{
		String output = "Super Model: \n";
		
		for (int k = 1; k <= NUM_LAYER; k++) {
			output = output + "weights[" + k + "]:\n";
			for (int j = 0; j < nodes_layer[k]; j++) {
				for (int i = 0; i < nodes_layer[k-1]; i++) {
					output = output + weights[k][nodes_layer[k-1]*j + i] + " ";
				}
				output = output + "\n";
			}
		}
		
		for (int k = 1; k <= NUM_LAYER; k++) {
			output = output + "bias[" + k + "]:\n";
			for (int j = 0; j < nodes_layer[k]; j++) {
				output = output + bh[k][j] + " ";
			}
			output = output + "\n";
		}
		
		output = output + "sub-models:\n";
		for (int i = 0; i < nModel; i++) {
			output = output + models.get(i).toString();
		}
		
		return output;
	}
	*/
	
	void work_update(int layer_ind){
		float[] x0 = new float[nodes_layer[layer_ind - 1]]; // data
		float[] h0 = new float[nodes_layer[layer_ind]];  // hidden
		float[] x1 = new float[nodes_layer[layer_ind - 1]];
		float[] h1 = new float[nodes_layer[layer_ind]];
		float[] inc_w = new float[nodes_layer[layer_ind-1]*nodes_layer[layer_ind]]; // previous increase of weights
		float[] inc_bv = new float[nodes_layer[layer_ind-1]];
		float[] inc_bh = new float[nodes_layer[layer_ind]];
		Arrays.fill(inc_w,0);
		Arrays.fill(inc_bv, 0);
		Arrays.fill(inc_bh, 0);

		for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
			x0[i] = sample_mem[layer_ind - 1][i];

		if (layer_ind != NUM_LAYER) { // normal layer        

			//perform real computation
			GlobalUtil1.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				sample_mem[layer_ind][j] = h0[j];

			for (int i = 0; i < nodes_layer[layer_ind]; i++) {
				if (rd.nextFloat() < h0[i])
					h0[i] = 1;
				else
					h0[i] = 0;
			}


			GlobalUtil1.sigm(x1, bv[layer_ind], weights[layer_ind], h0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

					GlobalUtil1.sigm(h1, bh[layer_ind], weights[layer_ind], x1,
							nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);

					for (int j = 0; j < nodes_layer[layer_ind]; j++)
						for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
							inc_w[j*nodes_layer[layer_ind-1] + i] = mu * inc_w[j*nodes_layer[layer_ind-1] + i]
									+ yita_w * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
							weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
									weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
											+inc_w[j*nodes_layer[layer_ind-1] + i];
						}

					for (int j = 0; j < nodes_layer[layer_ind]; j++) {
						inc_bh[j] = mu * inc_bh[j] + yita_bh*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
						bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[j];
					}

					for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
						inc_bv[i] = mu * inc_bv[i] + yita_bv*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
						bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[i];
					}
					// print the layer input data (just for testing)
		}
		else { // top layer
			//perform real computation
			for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
				h0[j] = bh[NUM_LAYER][j];
				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
					h0[j] = h0[j] + weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * x0[i];
			}

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				sample_mem[layer_ind][j] = h0[j];


			GlobalUtil1.sigm(x1, bv[layer_ind], weights[NUM_LAYER], h0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

			for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
				h1[j] = bh[NUM_LAYER][j];
				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
					h1[j] = h1[j] + weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * x1[i];
			}

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
					inc_w[j*nodes_layer[layer_ind-1] + i] = mu * inc_w[j*nodes_layer[layer_ind-1] + i]
							+ yita_wt * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
					weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
							weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
									+inc_w[j*nodes_layer[layer_ind-1] + i];
				}

			for (int j = 0; j < nodes_layer[layer_ind]; j++) {
				inc_bh[j] = mu * inc_bh[j] + yita_bht*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
				bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[j];
			}

			for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
				inc_bv[i] = mu * inc_bv[i] + yita_bvt*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
				bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[i];
			}
			// print the layer input data (just for testing)
		}
	}
}