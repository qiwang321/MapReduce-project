/*
 * Cloud9: A Hadoop toolkit for working with big data
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
// package model;
package model;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import edu.umd.cloud9.io.array.ArrayListOfFloatsWritable;
import edu.umd.cloud9.io.array.ArrayListOfIntsWritable;

/**
 * Representation of a graph node for PageRank. 
 *
 * @author Jimmy Lin
 * @author Michael Schatz
 */
public class ModelNode1 implements Writable {

	private ArrayListOfFloatsWritable[] weights = new ArrayListOfFloatsWritable[GlobalUtil1.NUM_LAYER+1];
  private ArrayListOfFloatsWritable[] bv = new ArrayListOfFloatsWritable[GlobalUtil1.NUM_LAYER+1];
	private ArrayListOfFloatsWritable[] bh = new ArrayListOfFloatsWritable[GlobalUtil1.NUM_LAYER+1];
	
	private ArrayListOfFloatsWritable weights_field;
	private ArrayListOfFloatsWritable bh_field;
	private ArrayListOfFloatsWritable bv_field;
	
	public ModelNode1() {
	}

	public ArrayListOfFloatsWritable[] getWeight() {
		return weights;
	}

  public ArrayListOfFloatsWritable[] getBH() {
    return bh;
  }

  public ArrayListOfFloatsWritable[] getBV() {
    return bv;
  }

	public void setWeight(ArrayListOfFloatsWritable[] weight, ArrayListOfFloatsWritable weights_field) {
		this.weights = weight;
		this.weights_field = weights_field;
	}
  public void setBH(ArrayListOfFloatsWritable[] bh, ArrayListOfFloatsWritable bh_field) {
    this.bh = bh;
    this.bh_field = bh_field;
  }
  public void setBV(ArrayListOfFloatsWritable[] bv, ArrayListOfFloatsWritable bv_field) {
    this.bv = bv;
    this.bv_field = bv_field;
  }

	

	/**
	 * Deserializes this object.
	 *
	 * @param in source for raw byte representation
	 */
	@Override
	public void readFields(DataInput in) throws IOException {
	  for (int i=1; i<GlobalUtil1.NUM_LAYER+1;i++) {
	    weights[i] = new ArrayListOfFloatsWritable();
	    weights[i].readFields(in);
	  }

    for (int i=1; i<GlobalUtil1.NUM_LAYER+1;i++) {
      bv[i] = new ArrayListOfFloatsWritable();
      bv[i].readFields(in);
    }
    
    for (int i=1; i<GlobalUtil1.NUM_LAYER+1;i++) {
      bh[i] = new ArrayListOfFloatsWritable();
      bh[i].readFields(in);    
    }
    
    weights_field = new ArrayListOfFloatsWritable();
    weights_field.readFields(in);
    bv_field = new ArrayListOfFloatsWritable();
    bv_field.readFields(in);
    bh_field = new ArrayListOfFloatsWritable();
    bh_field.readFields(in);
	}

	/**
	 * Serializes this object.
	 *
	 * @param out where to write the raw byte representation
	 */
	@Override
	public void write(DataOutput out) throws IOException {
    for (int i=1; i<GlobalUtil1.NUM_LAYER+1;i++) 
      weights[i].write(out);
		
    for (int i=1; i<GlobalUtil1.NUM_LAYER+1;i++) 
      bv[i].write(out);
		  
    for (int i=1; i<GlobalUtil1.NUM_LAYER+1;i++) 
      bh[i].write(out);
    
    weights_field.write(out);
    bv_field.write(out);
    bh_field.write(out);
	}

	@Override
	public String toString() {
		String output = "";
		for (int k = 1; k <= GlobalUtil1.NUM_LAYER; k++) {
			output = output + "weights[" + k + "]:\n";
			for (int j = 0; j < GlobalUtil1.nodes_layer[k]; j++) {
				for (int i = 0; i < GlobalUtil1.nodes_layer[k-1]; i++) {
					output = output + weights[k].get(GlobalUtil1.nodes_layer[k-1]*j + i) + " ";
				}
				output = output + "\n";
			}
		}
		for (int k = 1; k <= GlobalUtil1.NUM_LAYER; k++) {
			output = output + "bias[" + k + "]:\n";
			for (int j = 0; j < GlobalUtil1.nodes_layer[k]; j++) {
				output = output + bh[k].get(j) + " ";
			}
			output = output + "\n";
		}
		return output;
	}


  /**
   * Returns the serialized representation of this object as a byte array.
   *
   * @return byte array representing the serialized representation of this object
   * @throws IOException
   */
  public byte[] serialize() throws IOException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    DataOutputStream dataOut = new DataOutputStream(bytesOut);
    write(dataOut);

    return bytesOut.toByteArray();
  }

  /**
   * Creates object from a <code>DataInput</code>.
   *
   * @param in source for reading the serialized representation
   * @return newly-created object
   * @throws IOException
   */
  public static ModelNode1 create(DataInput in) throws IOException {
    ModelNode1 m = new ModelNode1();
    m.readFields(in);

    return m;
  }

  /**
   * Creates object from a byte array.
   *
   * @param bytes raw serialized representation
   * @return newly-created object
   * @throws IOException
   */
  public static ModelNode1 create(byte[] bytes) throws IOException {
    return create(new DataInputStream(new ByteArrayInputStream(bytes)));
  }
  
  
  
  public float[] sim(float[] data) {
  	
  	float[] res = new float[GlobalUtil1.NODES_INPUT];
  	float[] res_prev;
  	int n, m;
  	
  	int numWindows = GlobalUtil1.numWindows;
  	int window = GlobalUtil1.window;
  	int step = GlobalUtil1.step;
  	int imageSize = GlobalUtil1.imageSize;
  	int raw2d = GlobalUtil1.raw2d;
  	int new2d = GlobalUtil1.new2d;
  	
  	for (int l = 0; l < 3; l++)
  	for (int i = 0; i < numWindows; i++)
			for (int j = 0; j < numWindows; j++) {
			int ind_h = l*new2d + i*numWindows + j;
			res[ind_h] = -bh_field.get(ind_h);
			for (int s = 0; s < window; s++)
				for (int t = 0; t < window; t++) {
					int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
					int ind_w = ind_h*window*window + s*window+t;
					res[ind_h] = res[ind_h] - weights_field.get(ind_w) * data[ind_v];
				}
			res[ind_h] = 1.0f / (1.0f + (float)Math.exp(res[ind_h]));
		}
  	
  	res_prev = res;
  	
  	for (int i = 1; i <= GlobalUtil1.NUM_LAYER; i++) {
  		res = new float[GlobalUtil1.nodes_layer[i]];
  		n = GlobalUtil1.nodes_layer[i];
  		m = GlobalUtil1.nodes_layer[i-1];
  		GlobalUtil1.sigm(res, bh[i].getArray(), weights[i].getArray(), res_prev, n, m, true);
  		res_prev = res;
  	}
  	return res;
  }
}
