import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.Writable;



public class SuperModel implements Writable {
	private ArrayList<ModelNode> models;
	private int nModel;
	//private int nInput; // number of input nodes for the super layer
	private int nEach; // the number of outputs for each sub model
	
    private final Random rd = new Random();  
    private float[][] sample_mem = new float[GlobalUtil.SUPER_NUM+1][]; //space storing the MCMC samples
    private float[][] weights = new float[GlobalUtil.SUPER_NUM+1][]; //space storing the updating weights (first is not used)
    private float[][] bh = new float[GlobalUtil.SUPER_NUM+1][]; // hidden layer biases (rbm)
    private float[][] bv = new float[GlobalUtil.SUPER_NUM+1][]; // visible layer biases (rbm)  
    
    private int NUM_LAYER = GlobalUtil.SUPER_NUM;
    private int NODES_INPUT;    
    //private final int[] train_len = GlobalUtil.train_len; 
    //private final int[] test_len = GlobalUtil.test_len;
    private int[] nodes_layer = GlobalUtil.super_layer;
    
    // training parameters
    

    private float yita_w = GlobalUtil.yita_w, yita_bv = GlobalUtil.yita_bv, yita_bh = GlobalUtil.yita_bh,
    yita_wt = GlobalUtil.yita_wt, yita_bvt = GlobalUtil.yita_bvt, yita_bht = GlobalUtil.yita_bht; // learning rates
    private float mu = GlobalUtil.mu, reg = GlobalUtil.reg;
     
    
	
	public SuperModel() {
		models = new ArrayList<ModelNode>();
	}
	
	public SuperModel(ArrayList<ModelNode> m) {
		models = m;
		nModel = m.size();
		nEach = GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER];
		NODES_INPUT = nModel * nEach;
		nodes_layer = GlobalUtil.super_layer;
		nodes_layer[0] = NODES_INPUT;
		sample_mem[0] = new float[NODES_INPUT];
		for (int i = 1; i <= NUM_LAYER; i++) {
			weights[i] = new float[nodes_layer[i] * nodes_layer[i-1]];
			sample_mem[i] = new float[nodes_layer[i]];
		}
		
	}
	
	public void train(BufferedReader in) throws IOException {
		// readin the data
		float[] data = new float[GlobalUtil.NODES_INPUT]; // place to hold float data
		String line = in.readLine();
		while (line != null && line.length() > 0) {
			String[] tmp = line.split("\\s+");
			for (int i = 0; i < GlobalUtil.NODES_INPUT; i++) {
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
			
		}
	}

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
          GlobalUtil.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
                nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

          for (int j = 0; j < nodes_layer[layer_ind]; j++)
                sample_mem[layer_ind][j] = h0[j];
          
          for (int i = 0; i < nodes_layer[layer_ind]; i++) {
              if (rd.nextFloat() < h0[i])
                h0[i] = 1;
              else
                h0[i] = 0;
          }


          GlobalUtil.sigm(x1, bv[layer_ind], weights[layer_ind], h0,
                nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

          GlobalUtil.sigm(h1, bh[layer_ind], weights[layer_ind], x1,
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


            GlobalUtil.sigm(x1, bv[layer_ind], weights[NUM_LAYER], h0,
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
	@Override
	public void readFields(DataInput arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void write(DataOutput arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}
	
    void work_update(){
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
          GlobalUtil.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
                nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

          for (int j = 0; j < nodes_layer[layer_ind]; j++)
                sample_mem[layer_ind][j] = h0[j];
          
          for (int i = 0; i < nodes_layer[layer_ind]; i++) {
              if (rd.nextFloat() < h0[i])
                h0[i] = 1;
              else
                h0[i] = 0;
          }


          GlobalUtil.sigm(x1, bv[layer_ind], weights[layer_ind], h0,
                nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

          GlobalUtil.sigm(h1, bh[layer_ind], weights[layer_ind], x1,
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


            GlobalUtil.sigm(x1, bv[layer_ind], weights[NUM_LAYER], h0,
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